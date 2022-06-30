import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate
from models.vgg import vgg16
from wavetf import WaveTFFactory

# loss_mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

haar = WaveTFFactory.build('haar', dim=2)


class perpLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=None):
        super(perpLoss, self).__init__()
        # self.loss_mae = tf.keras.losses.MeanAbsoluteError(reduction=reduction)

    def call(self, y_true, y_pred):
        feats_yt = vgg16(y_true)
        feats_yp = vgg16(y_pred)
        # tf.print('feats0 ', feats_yp[0].shape[1], output_stream=sys.stdout)
        # tf.print('feats1 ', feats_yp[1].shape, output_stream=sys.stdout)
        # return self.loss_mae(feats_yt[0], feats_yp[0]) + self.loss_mae(feats_yt[1], feats_yp[1])
        return tf.reduce_mean(tf.abs(feats_yt[0] - feats_yp[0])) + tf.reduce_mean(tf.abs(feats_yt[1] - feats_yp[1]))


class haarLoss(tf.keras.losses.Loss):
    def __init__(self, reduction):
        # self.loss_mae = tf.keras.losses.MeanAbsoluteError(reduction=reduction)
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
        # tf.print('feats1 ', yp_haar.shape, output_stream=sys.stdout)
        return tf.reduce_mean(tf.abs(yt_haar - yp_haar))
        # return self.loss_mae(yt_haar, yp_haar)


class netLoss(tf.keras.losses.Loss):
    def __init__(self, mu, reduction):
        super(netLoss, self).__init__()
        self.mu = mu
        # self.loss_mse = tf.keras.losses.MeanSquaredError(reduction=reduction)
        self.loss_perp = perpLoss(reduction)
        self.loss_haar = haarLoss(reduction)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_true - y_pred)) + \
               0.01 * self.loss_perp.call(y_true, y_pred) + \
               0.1 * self.loss_haar.call(y_true, y_pred)
        # return self.loss_mse(y_true, y_pred) + self.mu * self.loss_perp.call(y_true, y_pred)
        # return self.loss_mse(y_true, y_pred) + self.mu * self.loss_perp.call(y_true, y_pred) + \
        #        self.mu * self.loss_haar.call(y_true, y_pred)

# @tf.function
# def loss_net(yt, yp):
#     loss = loss_mse(yt, yp) + mu * loss_perp(yt, yp)
#     return loss


# def loss_haar(yt, yp):
#     yt = haar.call(yt)
#     yp = haar.call(yp)
#     feats_yt = vgg16(yt[..., 1:])
#     feats_yp = vgg16(yp[..., 1:])
#     loss = loss_mae(feats_yt[0], feats_yp[0]) + loss_mae(feats_yt[1], feats_yp[1])
#     return loss


# def loss_perp(yt, yp):
#     yt = tf.tile(yt, tf.constant([1, 1, 1, 3], tf.int32))
#     yp = tf.tile(yp, tf.constant([1, 1, 1, 3], tf.int32))
#     feats_yt = vgg16(yt)
#     feats_yp = vgg16(yp)
#     loss = loss_mae(feats_yt[0], feats_yp[0]) + loss_mae(feats_yt[1], feats_yp[1])
#     return loss

# inp = tf.keras.layers.Input(shape=[224, 224, 3])
# haarLoss(tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE).call(inp, inp)
# netLoss(0.01)
# print()
# perpLoss().call(inp, inp)
