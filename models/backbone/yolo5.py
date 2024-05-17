import tensorflow as tf
from ..utils.conv_module import ConvBlock
from ..utils import COMMON_FACTORY
from pprint import pprint
import math
from keras_flops import get_flops

act = "relu"
BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3


class YOLOv5(tf.keras.Model):

    def __init__(self, config, out_indices=(4, 6), *args, **kwargs):
        super(YOLOv5, self).__init__(*args, **kwargs)
        self.config = config
        block_cfg = self.config.block_cfg
        self.out_indices = out_indices
        # building first layer
        self.stage_layers = []
        for i in range(2):
            name, coeff = block_cfg[i]
            if name == 'Conv':
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method="bn",
                                  activation="relu",
                                  name="stem")
            else:
                layer = COMMON_FACTORY.get(name)(*coeff, name="stem")
            self.stage_layers.append(layer)
        block_cfg = block_cfg[2:]
        for i, (name, coeff) in enumerate(block_cfg):
            if name == "Conv":
                layer = ConvBlock(filters=coeff[0],
                                  kernel_size=coeff[1],
                                  strides=coeff[2],
                                  use_bias=False,
                                  norm_method="bn",
                                  activation="relu")
            else:
                layer = COMMON_FACTORY.get(name)(*coeff)
            self.stage_layers.append(layer)

    @tf.function
    def call(self, x):
        output = []
        for i, layer in enumerate(self.stage_layers):
            x = layer(x)
            if i in self.out_indices:
                output.append(x)
        output.append(x)
        return tuple(output)


def yolov5net(config, input_shape, kernel_initializer=None):
    yolo = YOLOv5(config=config)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = yolo(image_inputs)
    # image_inputs = tf.keras.Input(shape=(384, 640, 3), name='image_inputs')
    # preds = yolo(image_inputs, training=False)
    # fully_models = tf.keras.Model(image_inputs, preds, name='fully')
    # fully_models.summary()
    # flops = get_flops(fully_models, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    # exit(1)
    return tf.keras.Model(image_inputs, fmaps)
