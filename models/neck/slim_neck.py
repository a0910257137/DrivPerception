import tensorflow as tf
from ..utils.conv_module import *

act = "relu"


class GBottleneck(tf.keras.layers.Layer):

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super(GBottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               norm_method="bn",
                               activation=act)
        self.conv2 = ConvBlock(filters=c2,
                               kernel_size=3,
                               strides=1,
                               norm_method="bn",
                               activation=act)
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        short_x = self.conv2(self.conv1(x)) if self.add else self.conv2(
            self.conv1(x))
        return x + short_x


class VoVGSCSP(tf.keras.layers.Layer):

    def __init__(self, c2, n=1, shortcut=True, e=0.5):
        super(VoVGSCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=act)
        self.conv2 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=act)
        self.gsblk = [GBottleneck(c_, c_, shortcut, e=e) for _ in range(n)]
        self.conv3 = ConvBlock(filters=c2,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=act)

    def __call__(self, x):
        x1 = self.conv1(x)
        for op in self.gsblk:
            x1 = op(x1)
        y = self.conv2(x)
        y = self.conv3(tf.concat([y, x1], axis=-1))
        return y


class C3(tf.keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c2, n=1, shortcut=True, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=act)
        self.conv2 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=act)
        self.conv3 = ConvBlock(filters=c2,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=act)
        self.m = [GBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)]

    def __call__(self, x):
        x1 = self.conv1(x)
        for layer in self.m:
            x1 = layer(x1)
        x2 = self.conv2(x)
        x = tf.concat([x1, x2], axis=-1)
        x = self.conv3(x)
        return x


class GSConv(tf.keras.layers.Layer):

    def __init__(self, c2, k=1, s=1, activate=act):
        # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
        super(GSConv, self).__init__()
        c_ = c2 // 2
        self.conv = ConvBlock(
            filters=c_,
            kernel_size=k,
            strides=s,
            use_bias=False,
            norm_method="bn",
            activation=activate,
            conv_mode="conv2d",
        )
        self.dw = ConvBlock(
            filters=None,
            kernel_size=5,
            strides=1,
            use_bias=False,
            norm_method="bn",
            activation=activate,
            conv_mode="dw_conv2d",
        )

    def __call__(self, x):
        x1 = self.conv(x)
        x2 = tf.concat([x1, self.dw(x1)], axis=-1)
        # suffle
        b, h, w, c = [tf.shape(x2)[i] for i in range(4)]
        b = 1
        x2 = tf.transpose(x2, [0, 3, 1, 2])
        b_c = b * c // 2
        y = tf.reshape(x2, [b_c, 2, h * w])
        y = tf.transpose(y, [1, 0, 2])
        y = tf.reshape(y, [2, -1, c // 2, h, w])
        y = tf.concat([y[0], y[1]], axis=1)
        y = tf.transpose(y, [0, 2, 3, 1])
        return y


class SlimNeck(tf.keras.Model):

    def __init__(self, config, end_level=-1, **kwargs):
        super(SlimNeck, self).__init__(**kwargs)

        config = config.neck
        self.in_channels = config.in_channels

        l_out_channels = config.l_out_channels
        self.num_outs = config.num_outs
        gs_out_channels = [128, 64, 64, 128]
        gs_kernel_size = [1, 1, 3, 3]
        gs_strides = [1, 1, 2, 2]
        # ---------------------------------
        c3_out_channels = [128, 64, 128, 256]
        start_level = config.start_level
        self.num_ins = len(self.in_channels)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
            assert self.num_outs == end_level - start_level
        self.start_level, self.end_level = start_level, end_level
        self.lateral_convs, self.gs_convs, self.vovgscsp_convs = [], [], []

        for i in range(4):
            self.gs_convs.append(
                GSConv(c2=gs_out_channels[i],
                       k=gs_kernel_size[i],
                       s=gs_strides[i]))
            self.vovgscsp_convs.append(
                VoVGSCSP(c2=c3_out_channels[i], n=1, shortcut=False, e=0.5))
            if i < 3:
                l_conv = ConvBlock(
                    filters=l_out_channels,
                    kernel_size=1,
                    strides=1,
                    kernel_initializer=tf.keras.initializers.HeNormal,
                    norm_method=None,
                    activation=None,
                    use_bias=False)
                self.lateral_convs.append(l_conv)
        self.resize = tf.image.resize
        self.last_vovgscp = VoVGSCSP(c2=256, n=1, shortcut=False, e=0.5)
        self.avgPool = tf.keras.layers.AveragePooling2D
        self.kernelSizes = [3, 5]
        self.high_lateral_conv = []
        for _ in range(len(self.kernelSizes)):
            self.high_lateral_conv.append(
                ConvBlock(filters=l_out_channels,
                          kernel_size=1,
                          strides=1,
                          use_bias=False,
                          norm_method=None,
                          activation=None))
        self.high_lateral_conv_attention = tf.keras.Sequential([
            ConvBlock(filters=l_out_channels,
                      kernel_size=1,
                      strides=1,
                      use_bias=True,
                      norm_method=None,
                      activation="relu"),
            ConvBlock(filters=3,
                      kernel_size=3,
                      strides=1,
                      use_bias=True,
                      norm_method=None,
                      activation=None)
        ])

    def call(self, x):
        assert len(x) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(x[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        adapPool_Features = []
        h, w = tf.shape(x[-1])[1], tf.shape(x[-1])[2]
        for i in range(len(self.kernelSizes)):
            adaptive_x = self.avgPool(pool_size=self.kernelSizes[i],
                                      strides=(1, 1),
                                      padding='valid')(x[-1])
            adaptive_x = tf.compat.v1.image.resize(images=adaptive_x,
                                                   size=(h, w),
                                                   method='bilinear',
                                                   align_corners=True)
            adaptive_x = self.high_lateral_conv[i](adaptive_x)
            adapPool_Features.append(adaptive_x)
        fusion_weights = tf.concat(adapPool_Features, axis=-1)
        fusion_weights = self.high_lateral_conv_attention(fusion_weights)
        fusion_weights = tf.math.sigmoid(fusion_weights)
        adap_pool_fusion = 0
        for i in range(len(self.kernelSizes)):
            adap_pool_fusion = adap_pool_fusion + tf.expand_dims(
                fusion_weights[..., i], axis=-1) * adapPool_Features[i]
        laterals[-1] = laterals[-1] + adap_pool_fusion
        # build top-down path
        used_backbone_levels = len(laterals)
        up_lateral_feats = []
        x = laterals[2]
        for i, j in zip(range(used_backbone_levels - 1, 0, -1), range(2)):
            prev_shape = tf.shape(laterals[i - 1])[1:3]
            x = self.gs_convs[j](x)
            up_lateral_feats.append(x)
            x = self.resize(images=x,
                            size=prev_shape,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            lateral_x = laterals[i - 1]
            x = tf.concat([lateral_x, x], axis=-1)
            if j == 0:
                da_encoder = x
            if j == 1:
                ll_encoder = x
            x = self.vovgscsp_convs[j](x)
        # build outputs
        # part 2: concat bottom-up path
        outs = []
        for i, j in zip(range(2, 4), range(1, -1, -1)):
            x = self.vovgscsp_convs[i](x)
            outs.append(x)
            x = self.gs_convs[i](x)
            later_x = up_lateral_feats[j]
            x = tf.concat([later_x, x], axis=-1)
        x = self.last_vovgscp(x)
        outs.append(x)
        return (outs, da_encoder, ll_encoder)
