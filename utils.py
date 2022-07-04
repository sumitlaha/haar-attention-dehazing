import tensorflow as tf


def psnr(y_true, y_pred):
    y_true = standardize(y_true, axes=[1, 2])
    y_pred = standardize(y_pred, axes=[1, 2])
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim(y_true, y_pred):
    y_true = standardize(y_true, axes=[1, 2])
    y_pred = standardize(y_pred, axes=[1, 2])
    return tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=11,
                         filter_sigma=1.5, k1=0.01, k2=0.03)


def standardize(data, axes=None, eps=1e-08):
    mean, var = tf.nn.moments(data, axes=axes, keepdims=True)
    z_score = (data - mean) / tf.sqrt(var + eps)
    data = tf.clip_by_value(z_score, clip_value_min=-3, clip_value_max=3)
    return (data + 3) / 6
