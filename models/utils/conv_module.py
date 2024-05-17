import math
import tensorflow as tf
import tensorflow_addons as tfa
from pprint import pprint


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias=True,
                 strides=1,
                 dilation_rate=1,
                 groups=1,
                 bias_initializer='zeros',
                 kernel_initializer=tf.keras.initializers.HeUniform(),
                 kernel_constraint=None,
                 activation='relu',
                 norm_method='bn',
                 conv_mode='conv2d',
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.norm_method = norm_method
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        self.strides = (strides, strides)
        self.dilation_rate = (dilation_rate, dilation_rate)
        # reg_layer = tf.keras.regularizers.L2(l2=1e-3)
        reg_layer = None
        if conv_mode == 'conv2d':
            self.conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                dilation_rate=self.dilation_rate,
                kernel_regularizer=reg_layer,
                padding='same',
                name='conv')
        elif conv_mode == 'sp_conv2d':
            self.conv = tf.keras.layers.SeparableConv2D(
                filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                use_bias=use_bias,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                depthwise_regularizer=reg_layer,
                pointwise_regularizer=reg_layer,
                depthwise_constraint=kernel_constraint,
                pointwise_constraint=kernel_constraint,
                bias_initializer=bias_initializer,
                dilation_rate=self.dilation_rate,
                padding='same',
                name='sp_conv')
        elif conv_mode == 'dw_conv2d':
            self.conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=self.strides,
                use_bias=use_bias,
                depthwise_initializer=kernel_initializer,
                depthwise_regularizer=reg_layer,
                bias_initializer=bias_initializer,
                padding='same',
                name='dw_conv')
        if norm_method == 'bn':
            self.norm = tf.keras.layers.BatchNormalization(name='bn')
        elif norm_method == 'range_bn':
            self.norm = RangeBN(filters=filters, name='range_bn')
        elif norm_method == 'gn':
            self.norm = tfa.layers.GroupNormalization(groups=groups, axis=-1)
        if activation in [
                'relu', 'swish', 'LeakyReLU', 'sigmoid', 'hard_sigmoid',
                'softmax'
        ]:
            self.act = tf.keras.layers.Activation(activation=activation,
                                                  name='act_' + activation)
        elif activation == 'silu':
            self.act = tf.nn.silu
        elif activation == 'mish':
            self.act = self.mish
        elif activation is None:
            pass
        else:
            raise Exception('Activation not support {}'.format(activation))

    @staticmethod
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    def call(self, input):

        output = self.conv(input)
        if self.norm_method == 'bn':
            output = self.norm(output)
        elif self.norm_method == 'range_bn':
            output = self.norm(output)
        if self.activation is not None:
            output = self.act(output)
        return output


class RangeBN(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 momentum=0.1,
                 num_chunks=16,
                 eps=1e-6,
                 name=None,
                 **kwargs):
        super(RangeBN, self).__init__(**kwargs)
        """
            Follow the reference https://arxiv.org/abs/1805.11046, Scalable Methods for 8-bit Training of Neural Networks
            With the range of min(x) - max(x) of  input distribution, making more tolerant to quantization.    
            Formula: x = (x - mu)/(C(n) - range(x - mu))
        """
        self.filters = filters
        self.momentum = momentum
        self.num_chunks = num_chunks
        self.eps = eps

    def build(self, input_shape):
        initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1)
        input_shape = tf.TensorShape(input_shape)
        self.weight = tf.Variable(initial_value=initializer(
            shape=(input_shape[-1], ), dtype=tf.dtypes.float32),
                                  name='weights',
                                  trainable=True)
        self.bias = self.add_weight(
            shape=(input_shape[-1], ),
            initializer=tf.keras.initializers.Constant(0.),
            name='bias',
            trainable=True)
        self.moving_mean = tf.Variable(initial_value=tf.constant(
            0., shape=(self.filters, )),
                                       shape=self.filters,
                                       dtype=tf.dtypes.float32,
                                       name='moving_mean',
                                       trainable=False)

        self.moving_variance = tf.Variable(initial_value=tf.constant(
            0., shape=(self.filters, )),
                                           shape=(self.filters, ),
                                           dtype=tf.dtypes.float32,
                                           name='moving_variance',
                                           trainable=False)
        super(RangeBN, self).build(input_shape)

    def call(self, inputs, training=False):
        if training:
            B, H, W, C = [tf.shape(inputs)[i] for i in range(4)]
            y = tf.transpose(inputs, (3, 0, 1, 2))
            y = tf.reshape(y,
                           [C, self.num_chunks, B * H * W // self.num_chunks])
            mean_max = tf.math.reduce_max(y, [-1])
            mean_max = tf.math.reduce_mean(mean_max, axis=-1)  # C
            mean_min = tf.math.reduce_min(y, [-1])
            mean_min = tf.math.reduce_mean(mean_min, axis=-1)  # C

            mean = tf.math.reduce_mean(tf.reshape(y, (C, -1)), axis=-1)  # C
            B, H, W, C = inputs.get_shape().as_list()
            if B is None:
                B = 1
            upper = (0.5 * 0.35) * (1. + (math.pi * math.log(4))**0.5)
            lower = ((2. * math.log(B * H * W // self.num_chunks))**0.5)
            scale_fix = upper / lower
            scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)

            self.moving_mean.assign(self.moving_mean * self.momentum +
                                    (mean * (1 - self.momentum)))
            self.moving_variance.assign(self.moving_variance * self.momentum +
                                        (scale * (1 - self.momentum)))
        else:
            mean = self.moving_mean
            scale = self.moving_variance
        out = (inputs - tf.reshape(mean, (1, 1, 1, self.filters))) * \
            tf.reshape(scale, (1, 1, 1,self.filters))
        if self.weight is not None:
            out = out * tf.reshape(self.weight, (1, 1, 1, self.filters))
        if self.bias is not None:
            out = out + tf.reshape(self.bias, (1, 1, 1, self.filters))
        return out


class DepthwiseSeparableConv(tf.keras.layers.Layer):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 strides=1,
                 groups=1,
                 activation='relu',
                 norm_method='bn',
                 conv_mode='conv2d',
                 kernel_initializer=tf.keras.initializers.HeUniform(),
                 **kwargs):
        super(DepthwiseSeparableConv, self).__init__(**kwargs)
        self.depth_conv = tf.keras.layers.Conv2D(
            filters=in_channel,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer,
            use_bias=False)
        self.gn1 = tfa.layers.GroupNormalization(groups=groups, axis=-1)
        # self.norm = tf.keras.layers.BatchNormalization(name='bn')
        self.act1 = tf.keras.layers.Activation(activation='relu')
        self.point_conv = ConvBlock(filters=out_channel,
                                    kernel_size=1,
                                    strides=1,
                                    groups=groups,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=False,
                                    norm_method=norm_method,
                                    activation='relu')

    def call(self, x):
        x = self.depth_conv(x)
        x = self.gn1(x)
        # x = self.norm(x)
        x = self.act1(x)
        x = self.point_conv(x)
        return x


class TransitionUp(tf.keras.layers.Layer):

    def call(self, inputs, up_method, skip=None, concat=True):
        _, h, w, _ = inputs.get_shape().as_list()
        h, w = h * 2, w * 2
        out = tf.image.resize(images=inputs,
                              size=(h, w),
                              preserve_aspect_ratio=False,
                              antialias=False,
                              method=up_method,
                              name='upsampling_%s' % self.name)
        if concat:
            out = tf.concat([out, skip], axis=-1)
        return out


class TransposeUp(tf.keras.layers.Layer):

    def __init__(self, filters, scale, norm_method="bn", activation="relu"):
        super().__init__()
        self.activation = activation
        self.norm_method = norm_method
        self.up_sample = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(2, 2),
            strides=(scale, scale),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            name='deconv_%s' % self.name)
        if self.norm_method == "bn":
            self.norm = tf.keras.layers.BatchNormalization(name='bn')
        elif norm_method == 'range_bn':
            self.norm = RangeBN(filters=filters, name='range_bn')
        if self.activation in ["relu", "silu"]:
            self.act = tf.keras.layers.Activation(activation=activation,
                                                  name='act_' + activation)

    def call(self, inputs, skip=None, concat=False, **kwargs):
        out = self.up_sample(inputs)
        if self.norm_method is not None:
            out = self.norm(out)
        elif self.norm_method == 'range_bn':
            output = self.norm(output)
        if self.activation is not None:
            out = self.act(out)
        if concat:
            out = tf.concat([out, skip], axis=-1)
        return out


class BottleNeck(tf.keras.layers.Layer):

    def __init__(self, input_chs, filters, kernel_size, e, s, is_squeeze, nl,
                 **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        """
            Bottleneck
            This function defines a basic bottleneck structure.
            # Arguments
                inputs: Tensor, input tensor of conv layer.
                filters: Integer, the dimensionality of the output space.
                kernel: An integer or tuple/list of 2 integers, specifying the
                    width and height of the 2D convolution window.
                e: Integer, expansion factor.
                    t is always applied to the input size.
                s: An integer or tuple/list of 2 integers,specifying the strides
                    of the convolution along the width and height.Can be a single
                    integer to specify the same value for all spatial dimensions.
                squeeze: Boolean, Whether to use the squeeze.
                nl: String, nonlinearity activation type.
            # Returns
                Output tensor.
        """
        self.alpha = 1.0
        self.nl = nl
        self.is_squeeze = is_squeeze
        tchannel = int(e)
        cchannel = int(self.alpha * filters)
        self.r = s == 1 and input_chs == filters
        self.bneck_conv = ConvBlock(filters=tchannel,
                                    kernel_size=1,
                                    strides=1,
                                    name=None,
                                    activation=self.nl,
                                    norm_method='bn')
        self.dw = DW(kernel_size=kernel_size, strides=s, n1=self.nl)
        if self.is_squeeze:
            self.se_blk = SE(tchannel)

        self.bneck_tran_conv = ConvBlock(filters=cchannel,
                                         kernel_size=1,
                                         strides=1,
                                         name='bneck_trans_conv',
                                         norm_method='bn',
                                         activation=None)
        if self.r:
            self.add = tf.keras.layers.Add()

    def call(self, x):
        inputs = x
        x = self.bneck_conv(x)
        x = self.dw(x)
        if self.is_squeeze:
            x = self.se_blk(x)
        x = self.bneck_tran_conv(x)
        if self.r:
            x = self.add([x, inputs])
        return x


class DW(tf.keras.layers.Layer):

    def __init__(self, kernel_size, strides, n1, **kwargs):
        super(DW, self).__init__(**kwargs)
        self.n1 = n1
        self.dw = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=(strides, strides),
            depth_multiplier=1,
            depthwise_initializer=tf.keras.initializers.HeUniform(),
            name='dw_conv',
            padding='same')

        self.bn = tf.keras.layers.BatchNormalization(name='dw_bn')
        self.relu = tf.keras.layers.ReLU(max_value=6.0)

    def call(self, x):
        x = self.dw(x)
        x = self.bn(x)
        if self.n1 == 'relu':
            # Relu 6
            x = self.relu(x)
        elif self.n1 == 'hard_swish':
            # hard swish
            x = x * self.relu(x + 3.0) / 6.0
        elif self.n1 == 'swish':
            x = tf.keras.activations.swish(x)
        return x


class SE(tf.keras.layers.Layer):

    def __init__(self, input_chs, **kwargs):

        super(SE, self).__init__(**kwargs)
        self.input_chs = input_chs
        self.glbap = tf.keras.layers.GlobalAvgPool2D()
        self.relu = tf.keras.layers.Dense(units=input_chs, activation='relu')
        self.hs = tf.keras.layers.Dense(units=input_chs,
                                        activation='hard_sigmoid')
        self.mlty = tf.keras.layers.Multiply()

    def call(self, x):
        inputs = x
        x = self.glbap(x)
        x = self.relu(x)
        x = self.hs(x)
        x = tf.reshape(x, [-1, 1, 1, self.input_chs])
        x = self.mlty([inputs, x])
        return x
