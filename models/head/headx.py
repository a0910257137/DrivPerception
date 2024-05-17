import tensorflow as tf
from ..utils import ConvBlock, TransposeUp
from ..utils import COMMON_FACTORY, BottleneckCSP
from pprint import pprint


class HeadX(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(HeadX, self).__init__(*args, **kwargs)
        self.head_cfg = config
        anchors = self.head_cfg.anchors
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.strides = tf.cast(self.head_cfg.strides,
                               tf.float32)  # strides computed during build
        self.det_block_cfg = config.det_block_cfg

        _, _, name, coeff = self.det_block_cfg[0]
        self.m = COMMON_FACTORY.get(name)(*coeff)
        self.da_block_cfg = config.da_block_cfg
        self.da_layers, self.da_save_layers = [], []
        for i, (f, n, name, coeff) in enumerate(self.da_block_cfg):
            if name == 'Conv':
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method=coeff[3],
                                  activation=coeff[4])
            elif name == 'Upsample':
                layer = tf.image.resize
            else:
                layer = COMMON_FACTORY.get(name)(*coeff)
            self.da_save_layers.extend(
                x % i for x in ([f] if isinstance(f, int) else f)
                if x != -1)  # append to savelist
            self.da_layers.append(layer)

        self.ll_block_cfg = config.ll_block_cfg
        self.ll_layers, self.ll_save_layers = [], []
        for i, (f, n, name, coeff) in enumerate(self.ll_block_cfg):
            if name == 'Conv':
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method=coeff[3],
                                  activation=coeff[4])
            elif name == 'TransposeUp':
                layer = TransposeUp(filters=coeff[0],
                                    scale=coeff[1],
                                    norm_method=coeff[3],
                                    activation=coeff[-1])

            elif name == 'BottleneckCSP':
                layer = BottleneckCSP(c1=coeff[0],
                                      c2=coeff[1],
                                      activation1=coeff[3],
                                      activation2=coeff[4])

            else:
                layer = COMMON_FACTORY.get(name)(*coeff)
            self.ll_save_layers.extend(
                x % i for x in ([f] if isinstance(f, int) else f)
                if x != -1)  # append to savelist
            self.ll_layers.append(layer)

    # @tf.function
    def call(self, x, training=False):
        y = []
        x, da_encoder_x, ll_encoder_x = x
        pred_branches = {"detection": self.m(x)}
        for i, (block, m) in enumerate(zip(self.da_block_cfg, self.da_layers)):
            f, _, name, coeff = block
            if name == 'Upsample':
                _, h, w, _ = [tf.shape(da_encoder_x)[i] for i in range(4)]
                upsampling_method = coeff[1]
                if upsampling_method == "nearest":
                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                da_encoder_x = m(images=da_encoder_x,
                                 size=(2 * h, 2 * w),
                                 method=method)
            else:
                da_encoder_x = m(da_encoder_x)
            y.append(da_encoder_x if i in self.da_save_layers else None)
        pred_branches["area_segmentation"] = da_encoder_x
        for i, (block, m) in enumerate(zip(self.ll_block_cfg, self.ll_layers)):
            ll_encoder_x = m(ll_encoder_x)
        pred_branches["lane_segmentation"] = ll_encoder_x
        return pred_branches
