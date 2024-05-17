import tensorflow as tf
import numpy as np
from pprint import pprint
from .loss_base import LossBase
import cv2
from .loss_functions import FocalLoss
from .core.postprocess import build_targets
from .core.general import bbox_iou, smooth_BCE
from .core.evaluate import SegmentationMetric


class AnchorLoss(LossBase):

    def __init__(self, config):
        self.max_obj_num = config.max_obj_num
        self.head_cfg = config.head
        self.config = config['loss']
        anchors = self.head_cfg.anchors
        self.num_classes = self.head_cfg.num_classes
        self.na = len(anchors[0]) // 2
        self.height, self.width = self.config.resize_size
        self.feat_strides = [(8, 8), (16, 16), (32, 32)]
        self.anchors = tf.reshape(tf.cast(
            anchors, tf.float32), (-1, 3, 2)) / tf.reshape(
                tf.cast(self.feat_strides, tf.float32), [-1, 1, 2])
        self.feat_sizes = tf.cast([(self.height / ss[0], self.width / ss[1])
                                   for ss in self.feat_strides], tf.float32)
        self.num_classes = self.config.num_classes
        self.batch_size = self.config.batch_size
        self.loss_dicts = self.get_loss(self.config)
        multi_head_lambda = self.config.multi_head_lambda
        if not multi_head_lambda:
            lambdas = [1.0 for _ in range(len(self.loss_dicts) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)
        self.lambdas = lambdas
        self.metric = SegmentationMetric(2)

    def get_loss(self, cfg):
        """
        get MultiHeadLoss
        Inputs:
            -cfg: configuration use the loss_name part or 
                function part(like regression classification)
            Returns:
            -loss: (MultiHeadLoss)
        """
        BCEcls = tf.nn.weighted_cross_entropy_with_logits
        BCEobj = tf.nn.weighted_cross_entropy_with_logits
        # BCEseg = tf.keras.losses.BinaryCrossentropy(
        #     from_logits=False,
        #     axis=-1,
        #     reduction=tf.keras.losses.Reduction.NONE)
        # Focal loss
        gamma = cfg.fl_gamma  # focal loss gamma
        if gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)
        loss_dict = {
            "BCEcls": {
                "func": BCEcls,
                "pos_weight": cfg.cls_pos_weight
            },
            "BCEobj": {
                "func": BCEobj,
                "pos_weight": cfg.obj_pos_weight
            },
            "BCEseg": {
                "func": tf.nn.weighted_cross_entropy_with_logits,
                "pos_weight": cfg.seg_pos_weight
            }
        }
        return loss_dict

    def build_loss(self, head_model, logits, targets, training):

        def true_n(lbox, tobj, nt, n, b, a, gj, gi):
            nt += n  # cumulative targets
            idxs = tf.concat(
                [b[:, None], a[:, None], gj[:, None], gi[:, None]], axis=-1)
            ps = tf.gather_nd(
                pi, idxs)  # prediction subset corresponding to targets
            # Regression
            pxy = tf.math.sigmoid(ps[:, :2]) * 2. - 0.5
            pwh = (tf.math.sigmoid(ps[:, 2:4]) * 2)**2 * anchors[i]
            pbox = tf.concat((pxy, pwh), axis=1)  # predicted box
            iou = bbox_iou(pbox, tbox[i], x1y1x2y2=False,
                           CIoU=True)  # iou(prediction, target)
            lbox += tf.math.reduce_mean(1.0 - iou)  # iou loss
            # Objectness
            gr = 1.0
            values = (1.0 - gr) + gr * tf.clip_by_value(
                iou, clip_value_min=0., clip_value_max=np.inf)
            tobj = tf.tensor_scatter_nd_update(tensor=tobj,
                                               indices=idxs,
                                               updates=values)
            return tobj, lbox

        def false_n(lbox, tobj):
            return tobj, lbox

        """
        Inputs:
            - head_fields: (list) output from each task head
            - head_targets: (list) ground-truth for each task head
            - model:

        Returns:
            - total_loss: sum of all the loss
            - head_losses: (tuple) contain all loss[loss1, loss2, ...]
        """

        tcls, tbox, indices, anchors = build_targets(self.anchors,
                                                     self.feat_sizes,
                                                     head_model,
                                                     logits["detection"],
                                                     targets['b_xywh'])

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)
        BCEcls, BCEobj, BCEseg = self.loss_dicts["BCEcls"], self.loss_dicts[
            "BCEobj"], self.loss_dicts["BCEseg"]
        # Calculate Losses
        nt = 0  # number of targets
        no = len(logits["detection"])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1
                                                   ]  # P3-5 or P3-6
        lcls, lbox, lobj, lseg_da, lseg_ll, liou_ll = 0., 0., 0., 0., 0., 0.

        # calculate detection loss
        for i, pi in enumerate(
                logits["detection"]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = tf.zeros_like(pi[..., 0])  # target obj
            n = tf.shape(b)[0]  # number of targets
            tobj, lbox = tf.cond(
                n > 0,
                true_fn=lambda: true_n(lbox, tobj, nt, n, b, a, gj, gi),
                false_fn=lambda: false_n(lbox, tobj))
            # if n:
            #     nt += n  # cumulative targets
            #     idxs = tf.concat(
            #         [b[:, None], a[:, None], gj[:, None], gi[:, None]],
            #         axis=-1)
            #     ps = tf.gather_nd(
            #         pi, idxs)  # prediction subset corresponding to targets
            #     # Regression
            #     pxy = tf.math.sigmoid(ps[:, :2]) * 2. - 0.5
            #     pwh = (tf.math.sigmoid(ps[:, 2:4]) * 2)**2 * anchors[i]
            #     pbox = tf.concat((pxy, pwh), axis=1)  # predicted box
            #     iou = bbox_iou(pbox, tbox[i], x1y1x2y2=False,
            #                    CIoU=True)  # iou(prediction, target)
            #     # lbox += (1.0 - iou).mean()  # iou loss
            #     losses['lbox'] += tf.math.reduce_mean(1.0 - iou)  # iou loss
            #     # Objectness
            #     gr = 1.0
            #     values = (1.0 - gr) + gr * tf.clip_by_value(
            #         iou, clip_value_min=0., clip_value_max=np.inf)
            #     tobj = tf.tensor_scatter_nd_update(tensor=tobj,
            #                                        indices=idxs,
            #                                        updates=values)
            # tobj[b, a, gj, gi] = (1.0 - gr) + gr * tf.clip_by_value(
            #         iou, clip_value_min=0., clip_value_max=np.inf)
            # Classification
            # if self.num_classes > 1:  # cls loss (only if multiple classes)
            #     t = tf.ones_like(ps[:, 5:]) * cn  # targets
            #     t[range(n), tcls[i]] = cp
            #     losses['lbox'] += BCEcls(ps[:, 5:], t)  # BCE
            lobj += tf.math.reduce_mean(BCEobj['func'](
                labels=tobj,
                logits=pi[..., 4],
                pos_weight=BCEseg["pos_weight"])) * balance[i]  # obj loss
        drive_area_seg_targets = tf.reshape(targets["b_masks"], [-1])
        drive_area_seg_predicts = tf.reshape(logits["area_segmentation"], [-1])

        lane_line_seg_targets = tf.reshape(targets["b_lanes"], [-1])
        lane_line_seg_predicts = tf.reshape(logits["lane_segmentation"], [-1])

        if isinstance(BCEseg, dict):
            lseg_da = BCEseg["func"](labels=drive_area_seg_targets,
                                     logits=drive_area_seg_predicts,
                                     pos_weight=BCEseg["pos_weight"])
            lseg_da = tf.math.reduce_mean(lseg_da)
            lseg_ll = BCEseg["func"](labels=lane_line_seg_targets,
                                     logits=lane_line_seg_predicts,
                                     pos_weight=BCEseg["pos_weight"])
            lseg_ll = tf.math.reduce_mean(lseg_ll)
            # lseg_da = BCEseg["func"](y_true=drive_area_seg_targets,
            #                          y_pred=drive_area_seg_predicts)
            # lseg_ll = BCEseg["func"](y_true=lane_line_seg_targets,
            #                          y_pred=lane_line_seg_predicts)

        pad_w, pad_h = targets['b_paddings'][0, 0], targets['b_paddings'][0, 1]
        # lane segmentation
        # lane_line_pred = tf.where(
        #     tf.math.reduce_max(logits["lane_segmentation"],
        #                        axis=-1) == logits["lane_segmentation"][..., 0],
        #     1, 0)
        # lane_line_gt = tf.where(
        #     tf.math.reduce_max(targets["b_lanes"],
        #                        axis=-1) == targets["b_lanes"][..., 0], 1, 0)
        lane_line_pred = tf.where(
            tf.math.reduce_max(logits["lane_segmentation"],
                               axis=-1) == logits["lane_segmentation"][..., 1],
            1, 0)
        lane_line_gt = tf.where(
            tf.math.reduce_max(targets["b_lanes"],
                               axis=-1) == targets["b_lanes"][..., 1], 1, 0)
        with tf.device('cpu'):
            IoU = tf.py_function(
                self.split_features,
                inp=[lane_line_pred, lane_line_gt, pad_w, pad_h],
                Tout=tf.float32)
        liou_ll = 1 - IoU
        s = 3 / no  # output count scaling
        lcls *= self.config.cls_gain * s * self.lambdas[0]
        lobj *= self.config.obj_gain * s * (1.4 if no == 4 else
                                            1.) * self.lambdas[1]
        lbox *= self.config.box_gain * s * self.lambdas[2]
        lseg_da *= self.config.da_seg_gain * self.lambdas[3]
        lseg_ll *= self.config.ll_seg_gain * self.lambdas[4]
        liou_ll *= self.config.ll_iou_gain * self.lambdas[5]
        if self.config.det_only or self.config.enc_det_only:
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll

        if self.config.seg_only or self.config.enc_seg_only:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox

        if self.config.lane_only:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da

        if self.config.drivable_only:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
        losses = {
            'total': lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll,
            'lcls': lcls,
            'lbox': lbox,
            'lobj': lobj,
            'lseg_da': lseg_da,
            'lseg_ll': lseg_ll,
            'liou_ll': liou_ll
        }
        return losses

    def split_features(self, lane_line_pred, lane_line_gt, pad_w, pad_h):
        lane_line_pred = lane_line_pred.numpy()
        lane_line_gt = lane_line_gt.numpy()
        pad_w, pad_h = int(pad_w), int(pad_h)
        lane_line_pred = lane_line_pred[:, pad_h:self.height - pad_h,
                                        pad_w:self.width - pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:self.height - pad_h,
                                    pad_w:self.width - pad_w]
        self.metric.reset()
        self.metric.addBatch(lane_line_pred, lane_line_gt)
        IoU = self.metric.IntersectionOverUnion()
        return IoU
