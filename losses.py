import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate

from models.vgg import vgg16
from wavetf import WaveTFFactory

haar = WaveTFFactory.build('haar', dim=2)


class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def call(self, y_true, y_pred):
        feats_yt = vgg16(y_true)
        feats_yp = vgg16(y_pred)
        return tf.reduce_mean(tf.abs(feats_yt[0] - feats_yp[0])) + tf.reduce_mean(tf.abs(feats_yt[1] - feats_yp[1]))


class HaarLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.add = Add()

    def call(self, y_true, y_pred):
        yt = haar.call(y_true)
        yp = haar.call(y_pred)
        yt_haar = Concatenate()([tf.expand_dims(self.add([yt[..., 3], yt[..., 6], yt[..., 9]]), -1),
                                 tf.expand_dims(self.add([yt[..., 4], yt[..., 7], yt[..., 10]]), -1),
                                 tf.expand_dims(self.add([yt[..., 5], yt[..., 8], yt[..., 11]]), -1)])
        yp_haar = Concatenate()([tf.expand_dims(self.add([yp[..., 3], yp[..., 6], yp[..., 9]]), -1),
                                 tf.expand_dims(self.add([yp[..., 4], yp[..., 7], yp[..., 10]]), -1),
                                 tf.expand_dims(self.add([yp[..., 5], yp[..., 8], yp[..., 11]]), -1)])
        return tf.reduce_mean(tf.abs(yt_haar - yp_haar))


class NetLoss(tf.keras.losses.Loss):
    def __init__(self, mu1, mu2):
        super(NetLoss, self).__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.loss_perp = PerceptualLoss()
        self.loss_haar = HaarLoss()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_true - y_pred)) + \
               self.mu1 * self.loss_perp.call(y_true, y_pred) + \
               self.mu2 * self.loss_haar.call(y_true, y_pred)
