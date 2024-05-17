import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2D
from .conv_module import ConvBlock


class SE(tf.keras.layers.Layer):
    """Squeeze-and-excitation layer."""

    def __init__(self, se_filters, output_filters, name=None):
        super().__init__(name=name)

        self._local_pooling = False
        self._act = self.act = tf.nn.silu
        self.conv_kernel_initializer = tf.keras.initializers.HeUniform()
        # Squeeze and Excitation layer.
        self._se_reduce = tf.keras.layers.Conv2D(
            se_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=self.conv_kernel_initializer,
            padding='same',
            use_bias=True,
            name='conv2d')
        self._se_expand = tf.keras.layers.Conv2D(
            output_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=self.conv_kernel_initializer,
            padding='same',
            use_bias=True,
            name='conv2d_1')

    def call(self, inputs):
        h_axis, w_axis = [1, 2]
        if self._local_pooling:
            se_tensor = tf.nn.avg_pool(
                inputs,
                ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
                strides=[1, 1, 1, 1],
                padding='VALID')
        else:
            se_tensor = tf.math.reduce_mean(inputs, [h_axis, w_axis],
                                            keepdims=True)

        se_tensor = self._se_expand(self._act(self._se_reduce(se_tensor)))
        return tf.sigmoid(se_tensor) * inputs


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

    def __init__(self,
                 input_filter,
                 output_filter,
                 se_ratio,
                 kernel_size,
                 expand_ratio,
                 stride,
                 local_pooling=False,
                 name=None):
        """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      mconfig: GlobalParams, a set of global parameters.
      name: layer name.
    """
        super().__init__(name=name)
        self._local_pooling = local_pooling
        self.se_ratio = se_ratio
        self._act = 'silu'
        self._has_se = (self.se_ratio is not None and 0 < self.se_ratio <= 1)
        self.endpoints = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.conv_dropout = None
        self.conv_kernel_initializer = tf.keras.initializers.HeUniform()
        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        """Builds block according to the arguments."""
        filters = self.input_filter * self.expand_ratio
        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        if self.expand_ratio != 1:
            self._expand_conv = ConvBlock(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=self.conv_kernel_initializer,
                activation='silu',
                use_bias=False,
                name='conv2d')

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = ConvBlock(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            kernel_initializer=self.conv_kernel_initializer,
            conv_mode='dw_conv2d',
            activation='silu',
            use_bias=False,
            name='depthwise_conv2d')

        #TODO: for debug
        if self._has_se:
            num_reduced_filters = max(1, int(self.input_filter * self.se_ratio))
            self._se = SE(num_reduced_filters, filters, name='se')

        else:
            self._se = None

        # Output phase.
        filters = self.output_filter

        self._project_conv = ConvBlock(
            filters=filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=self.conv_kernel_initializer,
            activation=None,
            norm_method='bn',
            use_bias=False,
            name='conv2d')

    def residual(self, inputs, x, training, survival_prob):
        if (self.strides == 1 and self.input_filters == self.output_filter):
            # Apply only if skip connection presents.
            if survival_prob:
                x = self.drop_connect(x, training, survival_prob)
            x = tf.add(x, inputs)
        return x

    def drop_connect(self, inputs, is_training, survival_prob):
        """Drop the entire conv with given survival probability."""
        # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
        if not is_training:
            return inputs

        # Compute tensor.
        batch_size = tf.shape(inputs)[0]
        random_tensor = survival_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1],
                                           dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        # Unlike conventional way that multiply survival_prob at test time, here we
        # divide survival_prob at training time, such that no addition compute is
        # needed at test time.
        output = inputs / survival_prob * binary_tensor
        return output

    def call(self, inputs, training, survival_prob=None):
        """Implementation of call().
    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
        x = self._depthwise_conv(x)
        if self.conv_dropout and self.expand_ratio > 1:
            x = tf.keras.layers.Dropout(self.conv_dropout)(x, training=training)
        if self._se:
            x = self._se(x)
        self.endpoints = {'expansion_output': x}
        x = self._project_conv(x)
        x = self.residual(inputs, x, training)
        return x


class FusedMBConvBlock(MBConvBlock):
    """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""

    def _build(self):
        """Builds block according to the arguments."""
        # pylint: disable=g-long-lambda

        # pylint: enable=g-long-lambda

        # block_args = self._block_args

        filters = self.input_filter * self.expand_ratio

        kernel_size = self.kernel_size
        if self.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=block_args.strides,
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False,
                name=get_conv_name())
            self._norm0 = utils.normalization(mconfig.bn_type,
                                              axis=self._channel_axis,
                                              momentum=mconfig.bn_momentum,
                                              epsilon=mconfig.bn_epsilon,
                                              groups=mconfig.gn_groups,
                                              name=get_norm_name())

        if self._has_se:
            num_reduced_filters = max(
                1, int(block_args.input_filters * block_args.se_ratio))
            self._se = SE(mconfig, num_reduced_filters, filters, name='se')
        else:
            self._se = None
        # Output phase:
        filters = block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1 if block_args.expand_ratio != 1 else kernel_size,
            strides=1 if block_args.expand_ratio != 1 else block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False,
            name=get_conv_name())
        self._norm1 = utils.normalization(mconfig.bn_type,
                                          axis=self._channel_axis,
                                          momentum=mconfig.bn_momentum,
                                          epsilon=mconfig.bn_epsilon,
                                          groups=mconfig.gn_groups,
                                          name=get_norm_name())

    def call(self, inputs, training, survival_prob=None):
        """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
        logging.info('Block %s  input shape: %s', self.name, inputs.shape)
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x), training=training))
        logging.info('Expand shape: %s', x.shape)

        self.endpoints = {'expansion_output': x}

        if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
            x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(x, training)

        if self._se:
            x = self._se(x)

        x = self._norm1(self._project_conv(x), training=training)
        if self._block_args.expand_ratio == 1:
            x = self._act(x)  # add act if no expansion.

        x = self.residual(inputs, x, training, survival_prob)
        logging.info('Project shape: %s', x.shape)
        return x


class ASPP(tf.keras.layers.Layer):

    def __init__(self, out_dims, **kwargs):
        super().__init__(**kwargs)
        conv_mode = 'sp_conv2d'
        self.conv_1 = ConvBlock(out_dims,
                                kernel_size=1,
                                strides=1,
                                use_bias=False,
                                norm_method="bn",
                                activation=None,
                                name="aspp_conv_1",
                                conv_mode=conv_mode)
        # dilate conv6
        self.aout_6 = ConvBlock(out_dims,
                                kernel_size=3,
                                strides=1,
                                dilation_rate=2,
                                use_bias=False,
                                norm_method="bn",
                                activation="relu",
                                name="aspp_aout_6",
                                conv_mode=conv_mode)
        # dilate conv12
        self.aout_12 = ConvBlock(out_dims,
                                 kernel_size=3,
                                 strides=1,
                                 dilation_rate=4,
                                 use_bias=False,
                                 norm_method="bn",
                                 activation="relu",
                                 name="aspp_aout_12",
                                 conv_mode=conv_mode)
        # dilate conv18
        self.aout_18 = ConvBlock(out_dims,
                                 kernel_size=3,
                                 strides=1,
                                 dilation_rate=8,
                                 use_bias=False,
                                 norm_method="bn",
                                 activation="relu",
                                 name="aspp_aout_18",
                                 conv_mode=conv_mode)

        self.img_pool_conv = ConvBlock(filters=out_dims,
                                       kernel_size=1,
                                       strides=1,
                                       use_bias=False,
                                       name='gap_conv_1')
        self.out_conv = ConvBlock(filters=out_dims,
                                  kernel_size=1,
                                  strides=1,
                                  use_bias=False,
                                  name='out_conv1')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        with tf.name_scope('aspp_layer'):
            _, h, w, _ = x.get_shape().as_list()
            out_1 = self.conv_1(x)
            # dilate conv6
            aout_6 = self.aout_6(x)
            # dilate conv12
            aout_12 = self.aout_12(x)
            # dilate conv18
            aout_18 = self.aout_18(x)

            # img pooling
            img_pool = tf.math.reduce_mean(x, [1, 2],
                                           name='global_average_pooling',
                                           keepdims=True)
            img_pool = self.img_pool_conv(img_pool)
            img_pool = tf.image.resize(images=img_pool,
                                       size=(h, w),
                                       preserve_aspect_ratio=False,
                                       antialias=False,
                                       method="bilinear")
            img_pool = self.bn(img_pool)
            concat_list = [out_1, aout_6, aout_12, aout_18, img_pool]
            aout = tf.concat(concat_list, axis=-1)
            out = self.out_conv(aout)
            return out
