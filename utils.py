import io

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms


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


def hist_match(img, ref):
    img = np.asarray(img * 255.0, dtype=np.uint8)
    ref = np.asarray(ref * 255.0, dtype=np.uint8)
    return match_histograms(img, ref, multichannel=True)


def ycbcr2rgb(yp, cbcr):
    yp = standardize(yp, axes=[1, 2])
    ycbcr = tf.concat([yp, cbcr], axis=-1)
    ycbcr = tf.cast(ycbcr * 255.0, tf.uint8)
    rgb = tfio.experimental.color.ycbcr_to_rgb(ycbcr)
    rgb = tf.cast(rgb, tf.float32) / 255.0
    return tf.clip_by_value(rgb, 0, 1)


def ycbcr2rgb2(y, yp, cbcr):
    yp = standardize(yp, axes=[1, 2])
    cbcr = tf.cast(cbcr * 255.0, tf.uint8)
    y_hm = tf.py_function(hist_match, inp=[yp, y], Tout=tf.uint8)
    ycbcr = tf.concat([y_hm, cbcr], axis=-1)
    rgb = tfio.experimental.color.ycbcr_to_rgb(ycbcr)
    rgb = tf.cast(rgb, tf.float32) / 255.0
    return tf.clip_by_value(rgb, 0, 1)


def ycbcr2rgb3(yp, cbcr):
    yp = standardize(yp, axes=[1, 2])
    y_cb = tf.py_function(hist_match, inp=[tf.expand_dims(cbcr[..., 0], -1), yp], Tout=tf.uint8)
    y_cr = tf.py_function(hist_match, inp=[tf.expand_dims(cbcr[..., 1], -1), yp], Tout=tf.uint8)
    ycbcr = tf.concat([tf.cast(yp * 255.0, tf.uint8), y_cb, y_cr], axis=-1)
    rgb = tfio.experimental.color.ycbcr_to_rgb(ycbcr)
    rgb = tf.cast(rgb, tf.float32) / 255.0
    return tf.clip_by_value(rgb, 0, 1)


def display_images(images, titles, samples):
    figure = plt.figure(figsize=(20, 12))
    sz = len(titles)
    for i in range(samples * sz):
        if i % sz == 1 or i % sz == 3:
            im1 = tf.expand_dims(images[i + 1], 0)
            im2 = tf.expand_dims(images[i], 0)
            im1 = tf.image.convert_image_dtype(im1, tf.float32)
            im2 = tf.image.convert_image_dtype(im2, tf.float32)
            val_psnr = psnr(im1, im2)
            val_ssim = ssim(im1, im2)
            title = titles[i % sz] + \
                    ' psnr:' + str(np.round(val_psnr.numpy(), 2)) + \
                    ', ssim:' + str(np.round(val_ssim.numpy(), 4))
        else:
            title = titles[i % sz]
        plt.subplot(samples, sz, i + 1, title=title)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i % sz == 3 or i % sz == 4:
            plt.imshow(images[i], vmin=0, vmax=1)
        else:
            plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            plt.colorbar()
    return figure


def display_images2(images, titles, samples):
    figure = plt.figure(figsize=(15, 12))
    sz = len(titles)
    for i in range(samples * sz):
        if i % sz == 1:
            im1 = tf.expand_dims(images[i + 1], 0)
            im2 = tf.expand_dims(images[i], 0)
            im1 = tf.image.convert_image_dtype(im1, tf.float32)
            im2 = tf.image.convert_image_dtype(im2, tf.float32)
            psnr = tf.image.psnr(im1, im2, max_val=1.0)
            ssim = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                                 filter_sigma=1.5, k1=0.01, k2=0.03)
            title = titles[i % sz] + \
                    ' psnr:' + str(np.round(psnr.numpy(), 2)) + \
                    ', ssim:' + str(np.round(ssim.numpy(), 4))
        else:
            title = titles[i % sz]
        plt.subplot(samples, sz, i + 1, title=title)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], vmin=0, vmax=1)
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def smart_resize(x, size, interpolation='bilinear'):
    """Resize images to a target size without aspect ratio distortion.
  TensorFlow image datasets typically yield images that have each a different
  size. However, these images need to be batched before they can be
  processed by Keras layers. To be batched, images need to share the same height
  and width.
  You could simply do:
  ```python
  size = (200, 200)
  ds = ds.map(lambda img: tf.image.resize(img, size))
  ```
  However, if you do this, you distort the aspect ratio of your images, since
  in general they do not all have the same aspect ratio as `size`. This is
  fine in many cases, but not always (e.g. for GANs this can be a problem).
  Note that passing the argument `preserve_aspect_ratio=True` to `resize`
  will preserve the aspect ratio, but at the cost of no longer respecting the
  provided target size. Because `tf.image.resize` doesn't crop images,
  your output images will still have different sizes.
  This calls for:
  ```python
  size = (200, 200)
  ds = ds.map(lambda img: smart_resize(img, size))
  ```
  Your output images will actually be `(200, 200)`, and will not be distorted.
  Instead, the parts of the image that do not fit within the target size
  get cropped out.
  The resizing process is:
  1. Take the largest centered crop of the image that has the same aspect ratio
  as the target size. For instance, if `size=(200, 200)` and the input image has
  size `(340, 500)`, we take a crop of `(340, 340)` centered along the width.
  2. Resize the cropped image to the target size. In the example above,
  we resize the `(340, 340)` crop to `(200, 200)`.
  Args:
    x: Input image or batch of images (as a tensor or NumPy array).
      Must be in format `(height, width, channels)` or
      `(batch_size, height, width, channels)`.
    size: Tuple of `(height, width)` integer. Target size.
    interpolation: String, interpolation to use for resizing.
      Defaults to `'bilinear'`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
  Returns:
    Array with shape `(size[0], size[1], channels)`. If the input image was a
    NumPy array, the output is a NumPy array, and if it was a TF tensor,
    the output is a TF tensor.
  """
    if len(size) != 2:
        raise ValueError('Expected `size` to be a tuple of 2 integers, '
                         f'but got: {size}.')
    img = tf.convert_to_tensor(x)
    if img.shape.rank is not None:
        if img.shape.rank < 3 or img.shape.rank > 4:
            raise ValueError(
                'Expected an image array with shape `(height, width, channels)`, '
                'or `(batch_size, height, width, channels)`, but '
                f'got input with incorrect rank, of shape {img.shape}.')
    shape = tf.shape(img)
    height, width = shape[-3], shape[-2]
    target_height, target_width = size
    if img.shape.rank is not None:
        static_num_channels = img.shape[-1]
    else:
        static_num_channels = None

    crop_height = tf.cast(
        tf.cast(width * target_height, 'float32') / target_width, 'int32')
    crop_width = tf.cast(
        tf.cast(height * target_width, 'float32') / target_height, 'int32')

    # Set back to input height / width if crop_height / crop_width is not smaller.
    crop_height = tf.minimum(height, crop_height)
    crop_width = tf.minimum(width, crop_width)

    crop_box_hstart = tf.cast(
        tf.cast(height - crop_height, 'float32') / 2, 'int32')
    crop_box_wstart = tf.cast(
        tf.cast(width - crop_width, 'float32') / 2, 'int32')

    if img.shape.rank == 4:
        crop_box_start = tf.stack([0, crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([-1, crop_height, crop_width, -1])
    else:
        crop_box_start = tf.stack([crop_box_hstart, crop_box_wstart, 0])
        crop_box_size = tf.stack([crop_height, crop_width, -1])

    img = tf.slice(img, crop_box_start, crop_box_size)
    img = tf.image.resize(
        images=img,
        size=size,
        method=interpolation)
    # Apparent bug in resize_images_v2 may cause shape to be lost
    if img.shape.rank is not None:
        if img.shape.rank == 4:
            img.set_shape((None, None, None, static_num_channels))
        if img.shape.rank == 3:
            img.set_shape((None, None, static_num_channels))
    if isinstance(x, np.ndarray):
        return img.numpy()
    return img
