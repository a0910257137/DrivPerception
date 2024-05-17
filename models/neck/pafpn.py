import tensorflow as tf
from ..utils.conv_module import *
from ..utils import COMMON_FACTORY


class PAFPN(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(PAFPN, self).__init__(**kwargs)
        neck_cfg = config.neck
        self.block_cfg = neck_cfg.block_cfg
        self.head_outputs = neck_cfg.head_output
        self.encoder_idx = neck_cfg.encoder_idx
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

    def call(self, lateral_x):
        x = lateral_x[-1]

        temp_feats, head_feats = [], []
        # bottom-to-top <-> top-to-bottom
        for i, (block,
                layer) in enumerate(zip(self.block_cfg, self.stage_layers)):
            name, coeff = block
            if name == 'Concat':
                if (i < 8):
                    lx = lateral_x[coeff[0]]
                else:
                    lx = temp_feats[coeff[0]]
                x = tf.concat([x, lx], axis=-1)
            elif name == 'Upsample':
                _, h, w, _ = [tf.shape(x)[i] for i in range(4)]
                upsampling_method = coeff[1]
                if upsampling_method == "nearest":
                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                x = layer(images=x, size=(2 * h, 2 * w), method=method)
            else:
                x = layer(x)
            if i == self.encoder_idx:
                encoder_feats = x
            if i in self.head_outputs:
                head_feats.append(x)
            temp_feats.append(x)

        return head_feats, encoder_feats
