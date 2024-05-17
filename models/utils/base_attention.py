import tensorflow as tf
import math
from .conv_module import ConvBlock

conv_mode = "sp_conv2d"


class SelfAttention(tf.keras.layers.Layer):

    def __init__(self, outdim, name, **kwargs):
        super().__init__(**kwargs)
        self.outdim = outdim
        self.qConv = ConvBlock(self.outdim,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               activation=None)
        self.kConv = ConvBlock(self.outdim,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               activation=None)
        self.vConv = ConvBlock(self.outdim,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               activation=None)
        self.kMaxpooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.vMaxpooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        # self.kAvgpooling = tf.keras.layers.AveragePooling2D(pool_size=2,
        #                                                     strides=2)
        # self.vAvgpooling = tf.keras.layers.AveragePooling2D(pool_size=2,
        #                                                     strides=2)
        self.cConv = ConvBlock(self.outdim,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               activation=None)
        self.is_placeholder = True
        self._name = name
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1, ),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=True)
        # self.alpha = self.add_weight(
        #     name='alpha_sp',
        #     shape=(1, ),
        #     initializer=tf.keras.initializers.Constant(1.),
        #     trainable=True)
        super(SelfAttention, self).build(input_shape)

    def hw_flatten(self, f_map):
        _, h, w, c = f_map.get_shape().as_list()
        return tf.reshape(f_map, shape=[-1, h * w, c])

    def call(self, inputs, **kwargs):
        # copied_inputs = inputs
        # inputs = inputs + self.alpha * pos

        query = self.qConv(inputs)
        key = self.kConv(inputs)
        key = self.kMaxpooling(key)
        # key = self.kAvgpooling(key)
        value = self.vConv(inputs)
        value = self.vMaxpooling(value)
        # value = self.vAvgpooling(value)
        query = self.hw_flatten(query)

        key = tf.transpose(self.hw_flatten(key), [0, 2, 1])

        energy = tf.matmul(query, key)

        atten_map = tf.nn.softmax(energy)

        atten_map = tf.transpose(atten_map, [0, 2, 1])

        value = self.hw_flatten(value)
        value = tf.transpose(value, [0, 2, 1])
        self_atten_map = tf.matmul(value, atten_map)

        _, h, w, c = inputs.get_shape().as_list()
        self_atten_map = tf.reshape(self_atten_map, shape=[-1, h, w, c])
        self_atten_map = self.cConv(self_atten_map)

        output = self.gamma * self_atten_map + inputs

        return output


class ChannelAttention(tf.keras.layers.Layer):

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.is_placeholder = True
        self._name = name
        super(ChannelAttention, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1, ),
            initializer=tf.keras.initializers.Constant(0.),
            trainable=True)
        super(ChannelAttention, self).build(input_shape)

    def hw_flatten(self, f_map):
        _, h, w, c = f_map.get_shape().as_list()
        return tf.reshape(f_map, shape=[-1, h * w, c])

    def call(self, inputs, **kwargs):
        query = tf.transpose(self.hw_flatten(inputs), [0, 2, 1])
        key = self.hw_flatten(inputs)
        energy = tf.matmul(query, key)
        atten_map = tf.nn.softmax(energy)
        value = self.hw_flatten(inputs)
        channel_atten_map = tf.matmul(value, atten_map)
        _, h, w, c = inputs.get_shape().as_list()
        channel_atten_map = tf.reshape(channel_atten_map, shape=[-1, h, w, c])
        output = self.gamma * channel_atten_map + inputs
        return output


class PositionEmbeddingSine(tf.keras.layers.Layer):
    """
    ![Visualization Positional Encoding](https://raw.githubusercontent.com/EmGarr/kerod/master/ressources/2d_pos_encoding.png)

    Arguments:
        output_dim: Dimension of the dense embedding.

    Call arguments:
        masks: A tensor of bool and shape [batch_size, w, h] where False means
            padding and True pixel from the image

    Call returns:
        tf.Tensor: The encoding a tensor of float and shape [batch_size, w, h, output_dim]
    """

    def __init__(self, output_dim=64, temperature=10000):
        super().__init__()
        self.temperature = temperature
        self.scale = 2 * math.pi
        if output_dim % 2 != 0:
            raise ValueError(
                "x an y embedding will be concatened to form a single vector "
                f"of shape output_dim. Please use a multiple of 2 (e.g {output_dim})"
            )
        self.dim = int(output_dim / 2)
        dim_t = tf.range(self.dim, dtype=tf.float32)
        self.dim_t = self.temperature**(2 * (dim_t // 2) / self.dim)

    def call(self, masks):
        """From a masks tensor compute the positional encoding

        Arguments:
            masks: A tensor of bool and shape [batch_size, w, h] where False means
                padding and True pixel from the image

        Returns:
            tf.Tensor: The encoding a tensor of float and shape [batch_size, w, h, output_dim]
        """
        masks = tf.cast(masks, self.compute_dtype)
        y_embed = tf.math.cumsum(masks, axis=1)
        x_embed = tf.math.cumsum(masks, axis=2)
        # Normalize x_embed and y_embed by the maximum values of the cumsum
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        pos_x = x_embed[..., None] / self.dim_t
        pos_y = y_embed[..., None] / self.dim_t
        pos_x = tf.stack([
            tf.math.sin(pos_x[..., 0::2]),
            tf.math.cos(pos_x[..., 1::2]),
        ],
                         axis=4)

        pos_y = tf.stack([
            tf.math.sin(pos_y[..., 0::2]),
            tf.math.cos(pos_y[..., 1::2]),
        ],
                         axis=4)

        batch_size, h, w = tf.shape(masks)[0], tf.shape(masks)[1], tf.shape(
            masks)[2]
        pos_x = tf.reshape(pos_x, (batch_size, h, w, -1))
        pos_y = tf.reshape(pos_y, (batch_size, h, w, -1))

        pos_emb = tf.concat([pos_y, pos_x], axis=-1)
        return pos_emb
