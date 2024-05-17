import tensorflow as tf
from ..utils.conv_module import ConvBlock
from pprint import pprint
import math
from keras_flops import get_flops

act = "relu"
BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3


def _make_divisible(channels, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_channels = max(min_value,
                       int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


class SGBlockExtra(tf.keras.layers.Layer):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 last=False,
                 name="sgb_extra"):
        super(SGBlockExtra, self).__init__(name=name)

        hidden_dim = int(inp * expand_ratio)
        self.conv_layers = []
        self.conv_layers.append(
            ConvBlock(filters=None,
                      kernel_size=1,
                      strides=stride,
                      norm_method="bn",
                      activation=act,
                      use_bias=False,
                      conv_mode="dw_conv2d"))
        self.conv_layers.append(
            ConvBlock(filters=hidden_dim,
                      kernel_size=1,
                      strides=1,
                      norm_method="bn",
                      activation=None,
                      use_bias=False,
                      conv_mode="pw_conv2d"))
        self.conv_layers.append(
            ConvBlock(filters=oup,
                      kernel_size=1,
                      strides=stride,
                      norm_method="bn",
                      activation=act,
                      use_bias=False,
                      conv_mode="dw_conv2d"))

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class SGBlock(tf.keras.layers.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 keep_3x3=False,
                 name='block'):
        super(SGBlock, self).__init__(name=name)
        assert stride in [1, 2]
        hidden_dim = in_channels // expand_ratio
        if hidden_dim < out_channels / 6.:
            hidden_dim = math.ceil(out_channels / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        self.conv_layers = []

        if expand_ratio == 2:
            # dw
            self.conv_layers.append(
                ConvBlock(filters=None,
                          kernel_size=3,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False,
                          conv_mode="dw_conv2d"))
            # pw-linear
            # pointwise reduction
            self.conv_layers.append(
                ConvBlock(filters=hidden_dim,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=None,
                          use_bias=False))
            # pw-linear
            # pointwise expansion
            self.conv_layers.append(
                ConvBlock(filters=out_channels,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False))
            # depthwise
            self.conv_layers.append(
                ConvBlock(filters=None,
                          kernel_size=3,
                          strides=stride,
                          norm_method="bn",
                          activation=None,
                          use_bias=False,
                          conv_mode="dw_conv2d"))

        elif in_channels != out_channels and stride == 1 and keep_3x3 == False:
            self.conv_layers.append(
                ConvBlock(filters=hidden_dim,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=None,
                          use_bias=False))
            self.conv_layers.append(
                ConvBlock(filters=out_channels,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False))
        elif in_channels != out_channels and stride == 2 and keep_3x3 == False:
            self.conv_layers.append(
                ConvBlock(filters=hidden_dim,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=None,
                          use_bias=False))
            self.conv_layers.append(
                ConvBlock(filters=out_channels,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False))
            self.conv_layers.append(
                ConvBlock(filters=None,
                          kernel_size=3,
                          strides=stride,
                          norm_method="bn",
                          activation=None,
                          use_bias=False,
                          conv_mode="dw_conv2d"))
        else:
            if keep_3x3 == False:
                self.identity = True
            # dw
            self.conv_layers.append(
                ConvBlock(filters=None,
                          kernel_size=3,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False,
                          conv_mode="dw_conv2d"))
            # pw
            self.conv_layers.append(
                ConvBlock(filters=hidden_dim,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=None,
                          use_bias=False,
                          conv_mode="dw_conv2d"))
            # pw
            self.conv_layers.append(
                ConvBlock(filters=out_channels,
                          kernel_size=1,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False,
                          conv_mode="dw_conv2d"))
            # dw
            self.conv_layers.append(
                ConvBlock(filters=out_channels,
                          kernel_size=3,
                          strides=1,
                          norm_method="bn",
                          activation=act,
                          use_bias=False,
                          conv_mode="dw_conv2d"))
        # self.residual = (in_channels == out_channels and stride == 1)

    def call(self, x, training=True):
        _x = x
        for layer in self.conv_layers:
            x = layer(x, training=training)
        if self.identity:
            return x + _x
        return x


class MobileNextNetModel(tf.keras.Model):

    def __init__(self, config, out_indices=(1, 4, 11, 19), *args, **kwargs):
        super(MobileNextNetModel, self).__init__(*args, **kwargs)
        self.config = config
        self.out_indices = out_indices
        width_mult = 1.
        # building first layer
        input_channel = _make_divisible(32 * width_mult,
                                        4 if width_mult == 0.1 else 8)
        self.stem, self.stage_layers = [], []
        self.stem.append(
            ConvBlock(filters=input_channel,
                      kernel_size=3,
                      strides=2,
                      use_bias=False,
                      norm_method="bn",
                      activation=act,
                      name='conv_stem'))

        self.config_blks = self.config.block_cfg
        # in_channels = stem_channels
        for i, (t, c, n, s) in enumerate(self.config_blks):
            output_channel = _make_divisible(c * width_mult,
                                             4 if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            self.stage_layers.append(
                SGBlock(input_channel, output_channel, s, t, n == 1
                        and s == 1))
            input_channel = output_channel
            for i in range(n - 1):
                self.stage_layers.append(
                    SGBlock(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        input_channel = output_channel
        output_channel = _make_divisible(input_channel, 4)

    @tf.function
    def call(self, x):
        output = []
        for stem_layer in self.stem:
            x = stem_layer(x)
        for i, layer in enumerate(self.stage_layers):
            x = layer(x)
            if i in self.out_indices:
                output.append(x)
        # output = output[::-1]
        return tuple(output)


def mobilenextnet(config, input_shape, kernel_initializer=None):
    mobilenextnet = MobileNextNetModel(config=config)
    image_inputs = tf.keras.Input(shape=input_shape, name='image_inputs')
    fmaps = mobilenextnet(image_inputs)
    # image_inputs = tf.keras.Input(shape=(384, 640, 3), name='image_inputs')
    # preds = mobilenextnet(image_inputs, training=False)
    # fully_models = tf.keras.Model(image_inputs, preds, name='fully')
    # fully_models.summary()
    # flops = get_flops(fully_models, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    # exit(1)
    return tf.keras.Model(image_inputs, fmaps)
