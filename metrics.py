import tensorflow as tf

from utils import standardize


class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='psnr'):
        super(PSNR, self).__init__()
        self.psnr = self.add_weight(name=name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = standardize(y_pred, axes=[1, 2])
        y_true = standardize(y_true, axes=[1, 2])
        values = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self.psnr.assign(tf.reduce_mean(values))

    def result(self):
        return self.psnr

    def reset_state(self):
        self.psnr.assign(0.0)


class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='ssim'):
        super(SSIM, self).__init__()
        self.ssim = self.add_weight(name=name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = standardize(y_pred, axes=[1, 2])
        y_true = standardize(y_true, axes=[1, 2])
        values = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                               filter_sigma=1.5, k1=0.01, k2=0.03)
        self.ssim.assign(tf.reduce_mean(values))

    def result(self):
        return self.ssim

    def reset_state(self):
        self.ssim.assign(0.0)
