import tensorflow as tf
from .conv_module import ConvBlock

conv_mode = "sp_conv2d"


class MP(tf.keras.layers.Layer):

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = tf.keras.layers.MaxPool2D(pool_size=(k, k),
                                           strides=k,
                                           padding='same')

    def __call__(self, x):
        return self.m(x)


class SPP(tf.keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c2, k=(5, 9, 13), **kwargs):
        super(SPP, self).__init__(**kwargs)
        c_ = c2 // 2
        self.conv1 = ConvBlock(c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu")
        self.conv2 = ConvBlock(c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu")
        self.m = []
        for x in k:
            self.m.append(
                tf.keras.layers.MaxPool2D(pool_size=(x, x),
                                          strides=1,
                                          padding='same'))

    def __call__(self, x):
        x = self.conv1(x)
        y1 = self.m[0](x)
        y2 = self.m[1](y1)
        y3 = self.m[2](y2)
        out = self.conv2(tf.concat([x, y1, y2, y3], axis=-1))
        return out


class SPPF(tf.keras.layers.Layer):

    def __init__(self, out_channels, name, **kwargs):
        super().__init__(**kwargs)
        c_ = out_channels // 2
        self.cv1 = ConvBlock(c_,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             activation="relu")
        self.cv2 = ConvBlock(out_channels,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             activation="relu")
        self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=(5, 5),
                                                    strides=1,
                                                    padding='same')

    def __call__(self, x):
        x = self.cv1(x)
        y1 = self.max_pool2d(x)
        y2 = self.max_pool2d(y1)
        y3 = self.max_pool2d(y2)
        out = self.cv2(tf.concat([x, y1, y2, y3], axis=-1))
        return out


##### cspnet #####


class SPPCSPC(tf.keras.layers.Layer):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvBlock(filters=c_,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")
        self.cv2 = ConvBlock(filters=c_,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")
        self.cv3 = ConvBlock(filters=c_,
                             kernel_size=3,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")
        self.cv4 = ConvBlock(filters=c_,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")
        self.m = [
            tf.keras.layers.MaxPool2D(pool_size=(x, x),
                                      strides=1,
                                      padding="same") for x in k
        ]
        self.cv5 = ConvBlock(filters=c_,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")
        self.cv6 = ConvBlock(filters=c_,
                             kernel_size=3,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")
        self.cv7 = ConvBlock(filters=c2,
                             kernel_size=1,
                             strides=1,
                             use_bias=False,
                             norm_method="bn",
                             activation="silu")

    def __call__(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(
            self.cv5(tf.concat([x1] + [m(x1) for m in self.m], axis=-1)))
        y2 = self.cv2(x)
        return self.cv7(tf.concat([y1, y2], axis=-1))


class Focus(tf.keras.layers.Layer):

    def __init__(self, c2, k=1, s=1, g=1, act='relu', **kwargs):
        super().__init__(**kwargs)
        # c2, k, s, p, g, act
        self.conv = ConvBlock(filters=c2,
                              kernel_size=k,
                              strides=s,
                              norm_method="bn",
                              use_bias=False,
                              activation=act)

    def __call__(self, x):
        x = self.conv(
            tf.concat([
                x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :],
                x[:, 1::2, 1::2, :]
            ],
                      axis=-1))
        return x


class Bottleneck(tf.keras.layers.Layer):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  #hidden channels
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu")

        self.conv2 = ConvBlock(filters=None,
                               kernel_size=3,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation="relu",
                               conv_mode="dw_conv2d")
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(
            self.conv1(x))


class SELayer(tf.keras.layers.Layer):

    def __init__(self, c, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

        self.fc = [
            tf.keras.layers.Dense(c // reduction, use_bias=False),
            tf.keras.layers.Activation(activation="relu"),
            tf.keras.layers.Dense(c, use_bias=False),
            tf.keras.layers.Activation(activation="sigmoid")
        ]

    def __call__(self, x):
        _, _, _, c = [tf.shape(x)[i] for i in range(4)]
        y = self.avg_pool(x)
        y = tf.reshape(y, [-1, c])
        for m in self.fc:
            y = m(y)
        y = tf.reshape(y, [-1, 1, 1, c])
        x += x * y
        return x


class BottleneckCSP(tf.keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5,
                 activation1="relu",
                 activation2="leaky_relu"
                 ):  # ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  #hidden channels
        self.conv1 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=activation1)
        self.conv2 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False)
        self.conv3 = ConvBlock(filters=c_,
                               kernel_size=1,
                               strides=1,
                               use_bias=False)
        self.conv4 = ConvBlock(filters=c2,
                               kernel_size=1,
                               strides=1,
                               use_bias=False,
                               norm_method="bn",
                               activation=activation1)
        self.bn = tf.keras.layers.BatchNormalization(name='bn')
        self.act = tf.keras.layers.Activation(activation=activation2,
                                              name='act_relu')
        self.m = []
        for _ in range(n):
            self.m.append(Bottleneck(c_, c_, shortcut=shortcut, g=g, e=1.0))

    def __call__(self, x):
        x0 = self.conv1(x)
        for layer in self.m:
            x0 = layer(x0)
        y1 = self.conv3(x0)
        y2 = self.conv2(x)
        out = self.conv4(self.act(self.bn(tf.concat([y1, y2], axis=-1))))
        return out


##### repvgg #####


class RepConv(tf.keras.layers.Layer):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        # self.in_channels = c1
        self.out_channels = c2
        assert k == 3

        if act is True:
            self.act = tf.nn.silu
        else:
            self.act = tf.identity

        if deploy:
            self.rbr_reparam = ConvBlock(filters=c2,
                                         kernel_size=k,
                                         strides=s,
                                         use_bias=True,
                                         norm_method=None,
                                         activation=None)

        else:
            self.rbr_identity = None
            self.rbr_dense = ConvBlock(filters=c2,
                                       kernel_size=k,
                                       strides=s,
                                       use_bias=False,
                                       norm_method="bn",
                                       activation=None)
            self.rbr_1x1 = ConvBlock(filters=c2,
                                     kernel_size=1,
                                     strides=s,
                                     use_bias=False,
                                     norm_method="bn",
                                     activation=None)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


##### yolor #####


class ImplicitA(tf.keras.layers.Layer):

    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        # Initialize all running statistics at 0.

        vals = tf.random.normal(shape=(1, 1, 1, int(self.channel)),
                                mean=self.mean,
                                stddev=self.std,
                                dtype=tf.dtypes.float32)

        self.implicit = tf.Variable(initial_value=vals,
                                    trainable=True,
                                    name='implicitA')

    def call(self, x):
        return self.implicit + x


class ImplicitM(tf.keras.layers.Layer):

    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        # Initialize all running statistics at 0.
        vals = tf.random.normal(shape=(1, 1, 1, int(self.channel)),
                                mean=self.mean,
                                stddev=self.std,
                                dtype=tf.dtypes.float32)
        self.implicit = tf.Variable(initial_value=vals,
                                    trainable=True,
                                    name='implicitM',
                                    dtype=tf.dtypes.float32)

    def call(self, x):
        return self.implicit * x


##### end of yolor #####
class IDetect(tf.keras.layers.Layer):

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.cast(anchors, tf.float32)
        self.anchors = tf.reshape(self.anchors, [self.nl, -1, 2])
        self.anchor_grid = tf.reshape(
            self.anchors, [self.nl, 1, -1, 1, 1, 2])  # shape(nl,1,na,1,1,2)
        # self.ch = [128., 256., 512.]
        self.ch = [128., 256., 256.]
        self.m = [
            ConvBlock(filters=self.no * self.na,
                      kernel_size=1,
                      strides=1,
                      use_bias=True,
                      norm_method=None,
                      activation=None) for _ in self.ch
        ]
        self.ia = [ImplicitA(x) for x in self.ch]
        self.im = [ImplicitM(self.no * self.na) for _ in self.ch]

    def __call__(self, x):
        output_feats = []
        for i in range(self.nl):
            lv_feat = x[i]
            lv_feat = self.m[i](self.ia[i](lv_feat))  # conv
            lv_feat = self.im[i](lv_feat)
            _, ny, nx, _ = [tf.shape(lv_feat)[j] for j in range(4)
                            ]  # x(bs,255,w,w) to x(bs,3,w,w,85)
            lv_feat = tf.reshape(lv_feat, (-1, ny, nx, self.na, self.no))
            # lv_feat = tf.reshape(lv_feat, [bs, ny * nx, self.na, self.no])
            # lv_feat = tf.transpose(lv_feat, (0, 2, 1, 3))
            # lv_feat = tf.reshape(lv_feat, (bs, self.na, ny, nx, self.no))
            output_feats.append(lv_feat)
        return output_feats


class Detect(tf.keras.layers.Layer):
    stride = None  # strides computed during build

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.cast(anchors, tf.float32)
        self.anchors = tf.reshape(self.anchors, [self.nl, -1, 2])
        self.anchor_grid = tf.reshape(
            self.anchors, [self.nl, 1, -1, 1, 1, 2])  # shape(nl,1,na,1,1,2)
        self.m = []
        for i in range(ch):
            self.m.append(
                ConvBlock(filters=self.no * self.na,
                          kernel_size=1,
                          strides=1,
                          use_bias=True,
                          norm_method=None,
                          activation=None))

    def __call__(self, x):
        return x
