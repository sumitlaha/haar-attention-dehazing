import matplotlib.pyplot as plt
import tensorflow as tf


def display_images(imgs, titles, samples):
    plt.figure(figsize=(20, 12))
    sz = len(titles)
    for i in range(samples * sz):
        title = titles[i % sz]
        plt.subplot(samples, sz, i + 1, title=title)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = tf.clip_by_value(imgs[i], 0, 1)
        plt.imshow(img, vmin=0, vmax=1)
    plt.show()


def standardize(data, axes=None, eps=1e-08):
    mean, var = tf.nn.moments(data, axes=axes, keepdims=True)
    z_score = (data - mean) / tf.sqrt(var + eps)
    data = tf.clip_by_value(z_score, clip_value_min=-3, clip_value_max=3)
    return (data + 3) / 6
