from pprint import pprint
from .base import Base
import numpy as np
import tensorflow as tf
import cv2


class DrivePostModel(tf.keras.Model):

    def __init__(self, pred_model, nc, n_objs, top_k_n, kp_thres,
                 nms_iou_thres, img_input_size, is_plot, *args, **kwargs):
        super(DrivePostModel, self).__init__(*args, **kwargs)
        self.pred_model = pred_model
        self.n_objs = n_objs
        self.top_k_n = top_k_n
        self.kp_thres = kp_thres
        self.nms_iou_thres = nms_iou_thres
        self.img_input_size = img_input_size
        self.is_plot = is_plot
        self.anchor_generator_strides = [(8, 8), (16, 16), (32, 32)]
        self.map_sizes = [(self.img_input_size[0] / s[0],
                           self.img_input_size[1] / s[1])
                          for s in self.anchor_generator_strides]
        self.grids = [
            self._meshgrid(size[1], size[0]) for size in self.map_sizes
        ]
        self.anchors = tf.cast([[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31],
                                [19, 50, 38, 81, 68, 157]], tf.int32)
        self.nl = 3
        self.anchor_grid = tf.reshape(self.anchors, [self.nl, 1, -1, 1, 1, 2])
        self.cls_out_channels = nc
        self.original_sizes = (720, 1280)
        self.base = Base()

    # @tf.function
    def call(self, x, training=False):
        imgs, self.ratios, self.paddings = x
        self.ratios = tf.tile(1 / self.ratios, [1, 2])
        self.paddings = tf.tile(self.paddings, [1, 2])

        batch_size = tf.shape(imgs)[0]
        # self.resize_ratio = tf.cast(origin_shapes / self.resize_shape,
        #                             tf.dtypes.float32)
        preds = self.pred_model(imgs, training=False)
        b_bboxes, da_seg_mask, ll_seg_mask, b_color_area = self._obj_detect(
            batch_size, preds['detection'], preds['area_segmentation'],
            preds['lane_segmentation'])
        return b_bboxes, da_seg_mask, ll_seg_mask, b_color_area

    # @tf.function
    def _obj_detect(self, batch_size, lv_detections, da_seg, ll_seg):
        b_outputs = -tf.ones(shape=(batch_size, self.n_objs,
                                    self.cls_out_channels, 5))
        tmp_boxes, tmp_scores, tmp_c, tmp_idx = [], [], [], []
        for i, detections in enumerate(lv_detections):
            # grid_shapes = self.grid[i].get_shape().as_list()
            det_shapes = detections.get_shape().as_list()
            bs, na, ny, nx, no = det_shapes
            y = tf.math.sigmoid(detections)
            xy = (y[..., :2] * 2. - 0.5 +
                  self.grids[i][None, None, ...]) * tf.cast(
                      self.anchor_generator_strides[i][0], tf.float32)  # xy
            wh = (y[..., 2:4] * 2)**2 * tf.cast(self.anchor_grid[i],
                                                tf.float32)  # wh
            y = tf.reshape(tf.concat([xy, wh, y[..., 4:]], axis=-1),
                           (batch_size, -1, no))
            boxes, scores, c, b_idx = self.get_bboxes(y,
                                                      conf_thres=0.25,
                                                      iou_thres=0.45,
                                                      classes=None,
                                                      agnostic=False)

            tmp_boxes.append(boxes)
            tmp_scores.append(scores)
            tmp_c.append(c)
            tmp_idx.append(b_idx)

        b_boxes = tf.concat(tmp_boxes, axis=0)
        b_scores = tf.concat(tmp_scores, axis=0)
        b_c = tf.concat(tmp_c, axis=0)
        b_idx = tf.concat(tmp_idx, axis=0)
        n = tf.shape(b_boxes)[0]
        idxs = tf.concat([
            b_idx,
            tf.range(n, dtype=tf.int32)[:, None],
            tf.cast(b_c, tf.int32)
        ],
                         axis=-1)
        b_boxes = tf.concat([b_boxes, b_scores], axis=-1)
        b_outputs = tf.tensor_scatter_nd_update(b_outputs, idxs, b_boxes)
        b_scores = b_outputs[..., -1]
        b_outputs = b_outputs[..., :-1]
        b_outputs -= self.paddings[:, None, None, :]

        b_outputs = tf.einsum('b n c d, b d -> b n c d', b_outputs,
                              self.ratios)

        nms_reuslt = tf.image.combined_non_max_suppression(
            b_outputs,
            b_scores,
            self.n_objs,
            self.n_objs,
            iou_threshold=self.nms_iou_thres,
            clip_boxes=False)
        box_results = tf.where(nms_reuslt[0] == -1., np.inf, nms_reuslt[0])
        box_results = tf.where((box_results - 1.) == -1., np.inf, box_results)
        b_bboxes = tf.concat(
            [box_results, nms_reuslt[1][..., None], nms_reuslt[2][..., None]],
            axis=-1)
        b_bboxes = tf.where(b_bboxes == -1., np.inf, b_bboxes)
        b_bboxes = tf.reshape(b_bboxes, [-1, self.n_objs, 6])

        h, w = self.img_input_size
        padding_w, padding_h = tf.cast(self.paddings[0, 0], tf.int32), tf.cast(
            self.paddings[0, 1], tf.int32)
        da_seg = da_seg[:, padding_h:(h - padding_h),
                        padding_w:(w - padding_w)]
        ll_seg = ll_seg[:, padding_h:(h - padding_h),
                        padding_w:(w - padding_w)]

        da_seg_mask = tf.where(
            tf.math.reduce_max(da_seg, axis=-1) == da_seg[..., 1], 1, 0)
        ll_seg_mask = tf.where(
            tf.math.reduce_max(ll_seg, axis=-1) == ll_seg[..., 1], 1, 0)
        idxs = tf.where(ll_seg_mask == 1)
        b_color_area = tf.zeros(shape=(batch_size, self.original_sizes[0],
                                       self.original_sizes[1], 3),
                                dtype=tf.float32)
        if self.is_plot:
            da_seg = tf.image.resize(da_seg,
                                     size=self.original_sizes,
                                     method='bilinear')
            ll_seg = tf.image.resize(ll_seg,
                                     size=self.original_sizes,
                                     method='bilinear')
            da_seg = tf.where(
                tf.math.reduce_max(da_seg, axis=-1) == da_seg[..., 1], 1, 0)
            ll_seg = tf.where(
                tf.math.reduce_max(ll_seg, axis=-1) == ll_seg[..., 1], 1, 0)
            _, h, w = [tf.shape(da_seg)[i] for i in range(3)]
            da_idxs = tf.where(da_seg == 1)
            da_clc = tf.tile(tf.constant([[0., 255., 0.]]),
                             [tf.shape(da_idxs)[0], 1])
            b_color_area = tf.tensor_scatter_nd_update(b_color_area, da_idxs,
                                                       da_clc)
            ll_idxs = tf.where(ll_seg == 1)
            ll_clc = tf.tile(tf.constant([[255., 0., 0.]]),
                             [tf.shape(ll_idxs)[0], 1])
            b_color_area = tf.tensor_scatter_nd_update(b_color_area, ll_idxs,
                                                       ll_clc)

        return b_bboxes, da_seg_mask, ll_seg_mask, b_color_area

    @staticmethod
    def _meshgrid(width, height):
        """Generate mesh grid of x and y.

        Args:
            x (tf.Tensor): Grids of x dimension.
            y (tf.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[tf.Tensor]: The mesh grids of x and y.
        """

        X, Y = tf.meshgrid(tf.range(0, width), tf.range(0, height))
        return tf.stack([X, Y], axis=-1)

    def get_bboxes(self,
                   prediction,
                   conf_thres=0.25,
                   iou_thres=0.45,
                   classes=None,
                   agnostic=False,
                   labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        nc = tf.shape(prediction)[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        idxs = tf.where(xc)
        prediction = tf.gather_nd(prediction, idxs)
        b_idx = tf.cast(idxs[:, :1], tf.int32)
        # Compute conf
        pred_confs = prediction[:,
                                5:] * prediction[:, 4:
                                                 5]  # conf = obj_conf * cls_conf
        box = self.base.xywh2xyxy(prediction[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = tf.concat((box[i], x[i, j + 5, None], j[:, None]), axis=1)
        else:  # best class only
            conf = tf.math.reduce_max(pred_confs, axis=-1, keepdims=True)
            j = tf.math.argmax(pred_confs, axis=-1)[:, None]
            idxs = tf.where(tf.reshape(conf, [-1]) > conf_thres)
            x = tf.concat([box, conf, tf.cast(j, tf.float32)], axis=-1)
            x = tf.gather_nd(x, idxs)
            b_idx = tf.gather_nd(b_idx, idxs)

        # Filter by class
        if classes is not None:
            mask = x[:, 5:6] == tf.constant(classes)
            mask = tf.math.reduce_any(mask)
        # Check shape
        n = tf.shape(x)[0]  # number of boxes
        if n > max_nms:
            idxs = tf.argsort(values=x[:, 4],
                              direction='DESCENDING',
                              name='sorting')[:max_nms]
            x = tf.gather_nd(x, idxs[:, None])  # sort by confidence
            b_idx = tf.gather_nd(b_idx, idxs[:, None])
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:,
                                        4:5]  # boxes (offset by class), scores
        return boxes, scores, c, b_idx
