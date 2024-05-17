import tensorflow as tf
from ..utils import ConvBlock
from ..utils import COMMON_FACTORY
from pprint import pprint


class Head(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(Head, self).__init__(*args, **kwargs)
        self.head_cfg = config
        self.strides = tf.cast(self.head_cfg.strides,
                               tf.float32)  # strides computed during build
        self.area_seg_idx = self.head_cfg.area_seg_idx
        self.lane_seg_idx = self.head_cfg.lane_seg_idx
        self.block_cfg = config.block_cfg
        self.num_classes = self.head_cfg.num_classes
        anchors = self.head_cfg.anchors
        self.in_channels = self.head_cfg.in_channels
        self.nc = self.num_classes  # number of classes
        self.no = self.num_classes + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.cast(anchors, tf.float32)
        self.anchors = tf.reshape(self.anchors, [self.nl, -1, 2])
        self.anchor_grid = tf.reshape(
            self.anchors, [self.nl, 1, -1, 1, 1, 2])  # shape(nl,1,na,1,1,2)

        self.det_in_channels = [128., 256., 512.]
        self.m = [
            ConvBlock(filters=self.no * self.na,
                      kernel_size=1,
                      strides=1,
                      use_bias=True,
                      norm_method=None,
                      activation=None,
                      name="Detect_{}".format(i))
            for i, _ in enumerate(self.in_channels)
        ]
        self.stage_layers = []
        for i, (name, coeff) in enumerate(self.block_cfg):
            if name == 'Conv':
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method="bn",
                                  activation="relu")
            elif name == 'Concat':
                layer = tf.concat
            elif name == 'Upsample':
                layer = tf.image.resize
            else:
                layer = COMMON_FACTORY.get(name)(*coeff)
            self.stage_layers.append(layer)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = tf.meshgrid(tf.range(0, nx), tf.range(0, ny))
        x = tf.stack((xv, yv), 2)
        x = tf.cast(tf.reshape(x, (1, 1, ny, nx, 2)), tf.float32)
        return x

    # @tf.function
    def call(self, x, training=False):
        x, encoder_x = x
        pred_branches = {}
        # deal with drivable area segmentation and lane line segmentation
        output_feats = []
        for i in range(self.nl):
            lv_feat = x[i]
            lv_feat = self.m[i](lv_feat)  # conv
            _, ny, nx, _ = [tf.shape(lv_feat)[j] for j in range(4)
                            ]  # x(bs,255,w,w) to x(bs,3,w,w,85)
            lv_feat = tf.reshape(lv_feat, (-1, ny, nx, self.na, self.no))
            # lv_feat = tf.reshape(lv_feat, [bs, ny * nx, self.na, self.no])
            # lv_feat = tf.transpose(lv_feat, (0, 2, 1, 3))
            # lv_feat = tf.reshape(lv_feat, (bs, self.na, ny, nx, self.no))
            output_feats.append(lv_feat)
        pred_branches["detection"] = output_feats
        copied_x = encoder_x
        for i, (block,
                layer) in enumerate(zip(self.block_cfg, self.stage_layers)):
            name, coeff = block
            if name == 'Upsample':
                _, h, w, _ = [tf.shape(encoder_x)[i] for i in range(4)]
                upsampling_method = coeff[1]
                if upsampling_method == "nearest":
                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                encoder_x = layer(images=encoder_x,
                                  size=(2 * h, 2 * w),
                                  method=method)
            else:
                encoder_x = layer(encoder_x)
            if i == self.area_seg_idx:
                pred_branches["area_segmentation"] = encoder_x
                encoder_x = copied_x
            if i == self.lane_seg_idx:
                pred_branches["lane_segmentation"] = encoder_x
        return pred_branches
