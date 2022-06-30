import sys

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
        # tf.print('\nPSNR: ', tf.reduce_mean(values), output_stream=sys.stdout)
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
        # tf.print('\nSSIM: ', tf.reduce_mean(values), output_stream=sys.stdout)
        self.ssim.assign(tf.reduce_mean(values))

    def result(self):
        return self.ssim

    def reset_state(self):
        self.ssim.assign(0.0)

# def mean2(x):
#     y = np.sum(x) / np.size(x)
#     return y
#
#
# def calc_en(im):
#     en = entropy(im * 255.0, base=2)
#     return en
#
#
# def calc_sd(im):
#     sd = np.std(im * 255.0, dtype=np.float64)  # Computing the standard deviation in float64 is more accurate
#     return sd
#
#
# def calc_ssim(im1, im2):
#     im1 = np.float32(im1 * 255.0)
#     im2 = np.float32(im2 * 255.0)
#     return ssim(im1, im2, multichannel=True)
#
#
# def calc_cc(im1, im2):
#     im1 = np.float32(im1 * 255.0)
#     im2 = np.float32(im2 * 255.0)
#     im1 = im1 - mean2(im1)
#     im2 = im2 - mean2(im2)
#     r = np.sum(im1 * im2) / np.sqrt(np.sum(im1 * im1) * np.sum(im2 * im2))
#     return r
#
#
# def calc_sf(im):
#     im = np.float32(im * 255.0)
#     rf = cf = 0
#     for i in range(im.shape[0]):
#         for j in range(1, im.shape[1]):
#             rf += (im[i, j] - im[i, j - 1]) ** 2
#     rf = np.sqrt(rf / np.float32(im.shape[0] * im.shape[1]))
#     for j in range(im.shape[1]):
#         for i in range(1, im.shape[0]):
#             cf += (im[i, j] - im[i - 1, j]) ** 2
#     cf = np.sqrt(cf / np.float32(im.shape[0] * im.shape[1]))
#     sf = np.sqrt(rf ** 2 + cf ** 2)
#     return np.average(sf)
