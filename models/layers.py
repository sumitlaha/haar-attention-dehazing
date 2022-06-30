import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, AveragePooling2D, Concatenate, Conv2D, Layer, \
    Dense
from tensorflow_addons.layers import InstanceNormalization

from wavetf import WaveTFFactory

tf.random.set_seed(1234)
INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=1234)


class ReflPad2D(Layer):
    def __init__(self, padding=(1, 1)):
        super(ReflPad2D, self).__init__()
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None, **kwargs):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor,
                      [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]],
                      'REFLECT')


class Cnv2D(Layer):
    def __init__(self, filters, kernel_size, norm='in', activation='relu'):
        super(Cnv2D, self).__init__()
        pad_size = int(kernel_size / 2)
        self.pad = ReflPad2D(padding=(pad_size, pad_size))
        self.cnv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1,
                          padding='valid', kernel_initializer=INIT, use_bias=False)
        self.norm = InstanceNormalization(axis=-1, center=True, scale=True,
                                          beta_initializer=INIT, gamma_initializer=INIT) if norm == 'in' else None
        self.act = Activation(activation) if activation == 'relu' else None

    def call(self, inputs, training=None, **kwargs):
        x1 = self.pad(inputs)
        x2 = self.cnv(x1)
        x3 = self.norm(x2, training=training) if self.norm is not None else x2
        x4 = self.act(x3) if self.act is not None else x3
        return x4


class ResBlock(Layer):
    def __init__(self, filters, kernel_size, use_cbam=False):
        super(ResBlock, self).__init__()
        self.use_cbam = use_cbam
        self.cnv0 = Cnv2D(filters, kernel_size)
        self.cnv1 = Cnv2D(filters, kernel_size, 'in', None)
        self.cbam = CBAM(filters)
        self.add = Add()
        self.act = Activation('relu')

    def call(self, inputs, training=None, **kwargs):
        x0 = inputs
        x1 = self.cnv0(x0, training=training)
        x2 = self.cnv0(x1, training=training)
        x3 = self.cnv1(x2, training=training)
        x4 = self.cbam(x3, training=training) if self.use_cbam else x3
        x5 = self.add([x0, x4])
        x6 = self.act(x5)
        return x6


class ChannelAttentionModule(Layer):
    def __init__(self, filters, reduction=8):
        super(ChannelAttentionModule, self).__init__()
        self.W0 = Dense(units=int(filters / reduction))
        self.W1 = Dense(units=int(filters))
        self.add = Add()
        self.act = Activation('relu')

    def call(self, inputs, training=None, **kwargs):
        x0 = inputs
        x1 = tf.reduce_mean(x0, [1, 2], True, 'avg_pool')  # Avg pooling
        x2 = tf.reduce_max(x0, [1, 2], True, 'max_pool')  # Max pooling
        x3 = self.W1(self.act(self.W0(x1)))
        x4 = self.W1(self.act(self.W0(x2)))
        x5 = self.add([x3, x4])
        x6 = self.act(x5)
        return x0 * x6


class SpatialAttentionModule(Layer):
    def __init__(self, filters):
        super(SpatialAttentionModule, self).__init__()
        self.concat = Concatenate()
        self.cnv = Cnv2D(filters, 7, None, None)
        self.sigma = Activation('sigmoid')

    def call(self, inputs, training=None, **kwargs):
        x0 = inputs
        x1 = tf.reduce_mean(x0, -1, True, 'avg_pool')  # Avg pooling
        x2 = tf.reduce_max(x0, -1, True, 'max_pool')  # Max pooling
        x3 = self.concat([x1, x2])
        x4 = self.cnv(x3, training=training)
        x5 = self.sigma(x4)
        return x0 * x5


class CBAM(Layer):
    def __init__(self, filters):
        super(CBAM, self).__init__()
        self.cam = ChannelAttentionModule(filters)
        self.sam = SpatialAttentionModule(filters)

    def call(self, inputs, training=None, **kwargs):
        x0 = self.cam(inputs)
        x1 = self.sam(x0, training=training)
        return x1


class MaxCoeff(Layer):
    def __init__(self):
        super(MaxCoeff, self).__init__()

    def call(self, inputs, training=None, **kwargs):
        x0 = tf.transpose(inputs[0], [0, 3, 1, 2])
        x1 = tf.transpose(inputs[1], [0, 3, 1, 2])
        x2 = tf.transpose(inputs[2], [0, 3, 1, 2])
        x3 = tf.where(tf.greater_equal(tf.abs(x0), tf.abs(x1)), x0, x1)
        x4 = tf.where(tf.greater_equal(tf.abs(x3), tf.abs(x2)), x3, x2)
        x5 = tf.transpose(x4, [0, 2, 3, 1])
        return x5


class MSBPlaneThresholding(Layer):
    def __init__(self):
        super(MSBPlaneThresholding, self).__init__()

    def call(self, inputs, training=None, **kwargs):
        x0 = tf.transpose(inputs, [0, 3, 1, 2])
        x1 = tf.where(tf.greater_equal(x0, 0.5), x0, 0)
        return tf.transpose(x1, [0, 2, 3, 1])


class EncoderModule2(Layer):
    def __init__(self, filters):
        super(EncoderModule2, self).__init__()
        self.msbT = MSBPlaneThresholding()
        self.haar = WaveTFFactory.build('haar', dim=2)
        self.cnv0 = Cnv2D(filters, 3)
        self.cnv1 = Cnv2D(filters * 2, 3, 'in', 'sigmoid')
        self.cam = ChannelAttentionModule(filters)
        self.sam = SpatialAttentionModule(filters)
        self.max = MaxCoeff()
        self.add = Add()
        # self.act = Activation('sigmoid')
        self.f = filters

    def call(self, inputs, training=None, **kwargs):
        x = self.msbT(inputs)
        x0 = self.haar.call(x)
        # x0 = self.haar.call(inputs)
        x1 = self.cnv0(x0[..., self.f:self.f * 2], training=training)
        x1 = self.sam(x1, training=training)
        x2 = self.cnv0(x0[..., self.f * 2:self.f * 3], training=training)
        x2 = self.sam(x2, training=training)
        x3 = self.cnv0(x0[..., self.f * 3:], training=training)
        x3 = self.sam(x3, training=training)
        x4 = self.max([x1, x2, x3])
        x5 = self.cnv0(x0[..., :self.f], training=training)
        x6 = self.cam(x5)
        x7 = self.add([x4, x6])
        x8 = self.cnv1(x7, training=training)
        # x9 = self.act(x8)
        return x8, x0[..., self.f:]


class DecoderModule2(Layer):
    def __init__(self, filters):
        super(DecoderModule2, self).__init__()
        self.ihaar = WaveTFFactory.build('haar', dim=2, inverse=True)
        self.cnv0 = Cnv2D(filters, 3)
        self.cnv1 = Cnv2D(filters, 3, 'in', 'sigmoid')
        self.pool = AveragePooling2D()
        # self.pool = MaxPool2D()
        self.res = ResBlock(filters=filters, kernel_size=3, use_cbam=True)

    def call(self, inputs, training=None, **kwargs):
        x0 = self.cnv0(inputs[0], training=training)
        x1 = self.pool(inputs[1])
        x2 = self.cnv1(Concatenate()([x0, x1]), training=training)
        x3 = self.ihaar.call(Concatenate()([x2, inputs[2]]))
        x4 = self.res(x3, training=training)
        return x4


class MainModule(Layer):
    def __init__(self, filters):
        super(MainModule, self).__init__()
        self.cnv0 = Cnv2D(5, 3)
        # self.cnv0 = Cnv2D(filters=int(filters/2), kernel_size=3)
        self.cnv1 = Cnv2D(filters, 3, 'in', 'sigmoid')
        # self.cnv2 = Cnv2D(filters=3, kernel_size=3, activation=None, norm=None)
        self.cnv2 = Cnv2D(3, 1, None, None)
        self.em1 = EncoderModule2(int(filters))
        self.em2 = EncoderModule2(int(filters * 2))
        self.em3 = EncoderModule2(int(filters * 4))
        self.em4 = EncoderModule2(int(filters * 8))
        self.dm0 = DecoderModule2(int(filters))
        self.dm1 = DecoderModule2(int(filters * 2))
        self.dm2 = DecoderModule2(int(filters * 4))
        self.dm3 = DecoderModule2(int(filters * 8))

    def call(self, inputs, training=None, **kwargs):
        # lvl E0
        e0_0 = self.cnv0(inputs, training=training)
        e0_1 = self.cnv1(e0_0, training=training)
        # lvl E1
        e1_0, e1_1 = self.em1(e0_1, training=training)
        # lvl E2
        e2_0, e2_1 = self.em2(e1_0, training=training)
        # lvl E3
        e3_0, e3_1 = self.em3(e2_0, training=training)
        # lvl E4
        e4_0, e4_1 = self.em4(e3_0, training=training)
        # lvl D3
        d3 = self.dm3([e4_0, e3_0, e4_1], training=training)
        # lvl D2
        d2 = self.dm2([d3, e2_0, e3_1], training=training)
        # lvl D1
        d1 = self.dm1([d2, e1_0, e2_1], training=training)
        # lvl D0
        d0_0 = self.dm0([d1, e0_1, e1_1], training=training)
        d0_1 = self.cnv2(d0_0, training=training)
        return d0_1
