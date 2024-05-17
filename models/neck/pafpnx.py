import tensorflow as tf
import numpy as np
from ..utils.conv_module import *
from ..utils import COMMON_FACTORY


class PAFPNX(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(PAFPNX, self).__init__(**kwargs)
        self.backbone_blk_cfg = config.backbone
        self.neck_blk_cfg = config.neck
        self.head_blk_cfg = config.head
        self.backbone_blk = self.backbone_blk_cfg.block_cfg
        self.backbone_output_idx = self.backbone_blk_cfg.output_idx
        self.neck_blk = self.neck_blk_cfg.block_cfg
        self.det_idxs = list(self.head_blk_cfg.det_block_cfg[0][0])
        self.num_backbone = len(self.backbone_blk)
        self.da_encoder_idx = self.neck_blk_cfg.da_encoder_idx
        self.ll_encoder_idx = self.neck_blk_cfg.ll_encoder_idx

        self.stage_layers = []
        for i, (f, _, name, coeff) in enumerate(self.neck_blk):
            if name == 'Conv':
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method="bn",
                                  activation="silu")
            elif name == 'Concat':
                layer = tf.concat
            elif name == 'Upsample':
                layer = tf.image.resize
            else:
                layer = COMMON_FACTORY.get(name)(*coeff)
            self.stage_layers.append(layer)
        self.save_layers = []
        for i, (f, n, name,
                coeff) in enumerate(self.backbone_blk + self.neck_blk):
            self.save_layers.extend(x % i
                                    for x in ([f] if isinstance(f, int) else f)
                                    if x != -1)  # append to savelist
        self.save_layers = sorted(self.save_layers)
        self.num_layers = len(self.backbone_blk)

    @tf.function
    def call(self, inputs):
        y = []
        inputs = list(inputs)
        x = inputs[0]
        for index in range(self.num_layers):
            if index in self.backbone_output_idx:
                y.append(inputs.pop())
            else:
                y.append(None)
        det_feats = []
        # bottom-to-top <-> top-to-bottom
        for i, (block, m) in enumerate(zip(self.neck_blk, self.stage_layers)):
            f, _, name, coeff = block
            if f != -1:  # if not from previous layer
                x = y[f] if isinstance(
                    f, int) else [x if j == -1 else y[j]
                                  for j in f]  # from earlier layers
            if name == "Concat":
                x = m(x, axis=-1)
            elif name == 'Upsample':
                _, h, w, _ = [tf.shape(x)[i] for i in range(4)]
                upsampling_method = coeff[1]
                if upsampling_method == "nearest":
                    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                x = m(images=x, size=(2 * h, 2 * w), method=method)
            else:
                x = m(x)
            i += self.num_backbone
            if i == self.da_encoder_idx:
                da_encoder = x
            elif i == self.ll_encoder_idx:
                ll_encoder = x
            elif i in self.det_idxs:
                det_feats.append(x)
            y.append(x if i in self.save_layers else None)  # save output
        return (det_feats, da_encoder, ll_encoder)
