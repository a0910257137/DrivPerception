import tensorflow as tf
from ..utils.conv_module import ConvBlock
from ..utils import COMMON_FACTORY
from pprint import pprint
from keras_flops import get_flops

BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3


class YOLOv7(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super(YOLOv7, self).__init__(*args, **kwargs)
        self.config = config
        self.backbone_cfg = self.config.backbone
        self.neck_cfg = self.config.neck
        self.backbone_blk = self.backbone_cfg.block_cfg
        self.output_idx = self.backbone_cfg.output_idx
        self.neck_blk = self.neck_cfg.block_cfg

        # building first layer
        self.stage_layers, self.save_layers = [], []
        gd = 1
        for i, (f, n, name, coeff) in enumerate(self.backbone_blk):
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if name == "Conv":
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method="bn",
                                  activation="silu")
            elif name == 'Concat':
                layer = tf.concat
            else:
                layer = COMMON_FACTORY.get(name)(*coeff)
            self.stage_layers.append(layer)
            self.save_layers.extend(x % i
                                    for x in ([f] if isinstance(f, int) else f)
                                    if x != -1)  # append to savelist
        self.save_layers = sorted(self.save_layers)

    @tf.function
    def call(self, x):
        y, outputs = [], []
        for i, (blk, m) in enumerate(zip(self.backbone_blk,
                                         self.stage_layers)):
            f, _, name, _ = blk
            if f != -1:  # if not from previous layer
                x = y[f] if isinstance(
                    f, int) else [x if j == -1 else y[j]
                                  for j in f]  # from earlier layers
            if name == "Concat":
                x = m(x, axis=-1)
            else:
                x = m(x)
            if i in self.output_idx:
                outputs.append(x)
            y.append(x if i in self.save_layers else None)  # save output
        return tuple(outputs[::-1])


def yolov7net(config, input_shape, kernel_initializer=None):
    yolo = YOLOv7(config=config)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = yolo(image_inputs)
    # image_inputs = tf.keras.Input(shape=(384, 640, 3), name='image_inputs')
    # preds = yolo(image_inputs, training=False)
    # fully_models = tf.keras.Model(image_inputs, preds, name='fully')
    # flops = get_flops(fully_models, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    # exit(1)
    return tf.keras.Model(image_inputs, fmaps)
