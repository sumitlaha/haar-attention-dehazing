import os
import random
from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

from utils import standardize

tf.random.set_seed(1234)

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 500
MUL_BETA = 5
IMG_WD = 320
IMG_HT = 240
SEED = tf.random.Generator.from_seed(1234, alg='philox').make_seeds(2)[0]


def read_npy(filename):
    return np.load(filename.numpy()).squeeze()


def create_synth_images(image, depth, mul_beta=1, invert_depth=False):
    # Standardize
    image = tf.cast(image, tf.float32)
    image /= 255.0

    depth = tf.cast(depth, tf.float32)
    depth = standardize(depth, axes=[0, 1])
    if invert_depth:
        depth = 1 - depth
    im_depth = tf.expand_dims(depth, axis=-1)

    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(SEED, num=2)

    # Initialize atmospheric light with random value
    atm_l = tf.random.stateless_uniform(shape=[], seed=new_seed[0, :], minval=0.7, maxval=1.0, dtype=tf.float32)
    # atm_l = tf.random.uniform(shape=[], minval=0.7, maxval=1.0, dtype=tf.float32)

    # Initialize scattering coefficient with random value
    beta = tf.random.stateless_uniform(shape=[mul_beta], seed=new_seed[1, :], minval=0.5, maxval=1.3, dtype=tf.float32)
    # beta = tf.random.uniform(shape=[mul_beta], minval=0.5, maxval=1.3, dtype=tf.float32)

    im_clr = []
    im_hazy = []
    for i in range(len(beta)):
        # Calculate transmission map
        t = tf.math.exp(-1 * beta[i] * im_depth)
        t = tf.tile(t, tf.constant([1, 1, 3], tf.int32))
        # Create synthetic hazy images
        j = tf.math.multiply(image, t) + atm_l * (1 - t)
        im_hazy.append(j)
        im_clr.append(image)

    return im_hazy, im_clr


def gen_hazy_nyudv2(elem):
    image, depth = elem['image'], elem['depth']
    im_hazy, im_clr = create_synth_images(image, depth, mul_beta=MUL_BETA)
    return im_hazy, im_clr


def gen_hazy_middlebury(image, depth):
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=3)
    depth = tf.io.read_file(depth)
    depth = tf.io.decode_png(depth, channels=1)
    depth = tf.squeeze(depth)
    im_hazy, im_clr = create_synth_images(image, depth, invert_depth=True)
    return im_hazy, im_clr


def gen_hazy_diode(image, depth):
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=3)
    depth = tf.py_function(read_npy, [depth], tf.float32)
    depth.set_shape([image.shape[0], image.shape[1]])
    im_hazy, im_clr = create_synth_images(image, depth, mul_beta=MUL_BETA)
    return im_hazy, im_clr


def gen_direct(im_hazy, im_clr):
    im_hazy = tf.io.read_file(im_hazy)
    im_hazy = tf.io.decode_png(im_hazy, channels=3)
    im_clr = tf.io.read_file(im_clr)
    im_clr = tf.io.decode_png(im_clr, channels=3)
    im_hazy = tf.cast(im_hazy, tf.float32)
    im_hazy /= 255.0
    im_clr = tf.cast(im_clr, tf.float32)
    im_clr /= 255.0
    return im_hazy, im_clr


def gen_direct2(im_hazy):
    im_hazy = tf.io.read_file(im_hazy)
    im_hazy = tf.io.decode_png(im_hazy, channels=3)
    im_hazy = tf.cast(im_hazy, tf.float32)
    im_hazy /= 255.0
    return im_hazy


def load_nyudv2(size):
    ds = tfds.load('nyu_depth_v2', split=tfds.core.ReadInstruction('train', from_=0, to=size, unit='abs'),
                   shuffle_files=False)
    ds = ds.map(gen_hazy_nyudv2, num_parallel_calls=AUTOTUNE)
    ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    return ds


def load_middlebury():
    data_dir = '../../../datasets/middlebury'
    files = sorted([f for f in glob(join(data_dir, '**/*.png'))])
    dep_list = sorted(files[0::4] + files[1::4])
    img_list = sorted(files[2::4] + files[3::4])
    ds = tf.data.Dataset.from_tensor_slices((img_list, dep_list))
    ds = ds.map(gen_hazy_middlebury, num_parallel_calls=AUTOTUNE)
    ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    return ds


def load_diode(size=None):
    data_dir = '../../../datasets/DIODE/train/indoors'
    image_list = [os.path.join(root, name)
                  for root, _, files in os.walk(data_dir)
                  for name in files
                  if name.endswith('.png')]
    if size:
        random.seed(1234)
        image_list = random.sample(image_list, size)
    depth_list = [path.replace('.png', '_depth.npy')
                  for path in image_list]
    img_list = sorted(image_list)
    dep_list = sorted(depth_list)

    ds = tf.data.Dataset.from_tensor_slices((img_list, dep_list))
    ds = ds.map(gen_hazy_diode, num_parallel_calls=AUTOTUNE)
    ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    return ds


def load_sots_indoor():
    data_dir = '../../../datasets/RESIDE/SOTS/indoor'
    list_hz = sorted([f for f in glob(join(data_dir, 'hazy/*.png'))])
    list_clr = [path.split('_')[0].replace('hazy', 'gt') + '.png'
                for path in list_hz]
    ds = tf.data.Dataset.from_tensor_slices((list_hz, list_clr))
    ds = ds.map(gen_direct, num_parallel_calls=AUTOTUNE)
    return ds


def load_sots_outdoor():
    data_dir = '../../../datasets/RESIDE/SOTS/outdoor'
    list_hz = sorted([f for f in glob(join(data_dir, 'hazy/*.jpg'))])
    list_clr = [path.split('_')[0].replace('hazy', 'gt') + '.png'
                for path in list_hz]
    ds = tf.data.Dataset.from_tensor_slices((list_hz, list_clr))
    ds = ds.map(gen_direct, num_parallel_calls=AUTOTUNE)
    return ds


def load_dense_haze():
    data_dir = '../../../datasets/Dense_Haze_NTIRE19'
    list_hz = sorted([f for f in glob(join(data_dir, 'hazy/*.png'))])
    list_clr = [path.replace('hazy', 'GT')
                for path in list_hz]
    ds = tf.data.Dataset.from_tensor_slices((list_hz, list_clr))
    ds = ds.map(gen_direct, num_parallel_calls=AUTOTUNE)
    return ds


def load_nh_haze():
    data_dir = '../../../datasets/NH-HAZE'
    list_hz = sorted([f for f in glob(join(data_dir, 'hazy/*.png'))])
    list_clr = [path.replace('hazy', 'GT')
                for path in list_hz]
    ds = tf.data.Dataset.from_tensor_slices((list_hz, list_clr))
    ds = ds.map(gen_direct, num_parallel_calls=AUTOTUNE)
    return ds


def load_rain12():
    data_dir = '../../../datasets/rain12'
    list_clear = sorted([f for f in glob(join(data_dir, '*_GT.png'))])
    list_rain = sorted([f for f in glob(join(data_dir, '*_in.png'))])
    ds = tf.data.Dataset.from_tensor_slices((list_rain, list_clear))
    ds = ds.map(gen_direct, num_parallel_calls=AUTOTUNE)
    return ds


def load_colorlines():
    data_dir = '../../../datasets/fattal_colorlines'
    list_hazy = sorted([f for f in glob(join(data_dir, '*.png'))])
    ds = tf.data.Dataset.from_tensor_slices(list_hazy)
    ds = ds.map(gen_direct2, num_parallel_calls=AUTOTUNE)
    return ds


def resize_and_rescale(im_hzy, im_clr):
    imgs = tf.concat([im_hzy, im_clr], axis=-1)
    imgs = tf.keras.preprocessing.image.smart_resize(imgs, size=[IMG_HT, IMG_WD])
    imgs = tf.clip_by_value(imgs, 0, 1)
    im_hzy = imgs[..., :3]
    im_clr = imgs[..., 3:]
    return im_hzy, im_clr


def resize_and_rescale2(im_hzy):
    imgs = im_hzy
    imgs = tf.keras.preprocessing.image.smart_resize(imgs, size=[IMG_HT, IMG_WD])
    imgs = tf.clip_by_value(imgs, 0, 1)
    im_hzy = imgs
    return im_hzy


def rgb2ycbcr(im_hzy, im_clr):
    ycbcr_hzy = tfio.experimental.color.rgb_to_ycbcr(tf.cast(im_hzy * 255.0, tf.uint8))
    ycbcr_clr = tfio.experimental.color.rgb_to_ycbcr(tf.cast(im_clr * 255.0, tf.uint8))
    ycbcr_hzy = tf.cast(ycbcr_hzy, tf.float32) / 255.0
    ycbcr_clr = tf.cast(ycbcr_clr, tf.float32) / 255.0
    ycbcr_hzy = tf.clip_by_value(ycbcr_hzy, 0, 1)
    ycbcr_clr = tf.clip_by_value(ycbcr_clr, 0, 1)
    y_hzy = tf.expand_dims(ycbcr_hzy[..., 0], -1)
    y_clr = tf.expand_dims(ycbcr_clr[..., 0], -1)
    ycbcr = tf.concat([ycbcr_hzy, ycbcr_clr], axis=-1)
    rgb = tf.concat([im_hzy, im_clr], axis=-1)
    return y_hzy, y_clr, ycbcr, rgb


def rgb2ycbcr2(im_hzy):
    ycbcr_hzy = tfio.experimental.color.rgb_to_ycbcr(tf.cast(im_hzy * 255.0, tf.uint8))
    ycbcr_hzy = tf.cast(ycbcr_hzy, tf.float32) / 255.0
    ycbcr_hzy = tf.clip_by_value(ycbcr_hzy, 0, 1)
    y_hzy = tf.expand_dims(ycbcr_hzy[..., 0], -1)
    ycbcr = ycbcr_hzy
    rgb = im_hzy
    return y_hzy, ycbcr, rgb


def augment(data, seed):
    im_hzy, im_clr = data
    imgs = tf.concat([im_hzy, im_clr], axis=-1)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=3)
    imgs = tf.image.stateless_random_crop(imgs, size=[IMG_HT - 10, IMG_WD - 10, 6], seed=new_seed[0, :])
    imgs = tf.keras.preprocessing.image.smart_resize(imgs, size=[IMG_HT, IMG_WD])
    imgs = tf.image.stateless_random_flip_left_right(imgs, new_seed[1, :])
    imgs = tf.image.stateless_random_flip_up_down(imgs, new_seed[2, :])
    imgs = tf.clip_by_value(imgs, 0, 1)
    im_hzy = imgs[..., :3]
    im_clr = imgs[..., 3:]
    return im_hzy, im_clr


# A wrapper function for updating seeds
def f(x, y):
    m, n = augment((x, y), SEED)
    im_hazy = [x, m]
    im_clr = [y, n]
    return im_hazy, im_clr


def prepare(ds, batch_size, shuffle=False, aug=False, luma=False):
    # Resize and rescale all datasets
    ds = ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)

    # Use data augmentation only on the training set
    if aug:
        ds = ds.map(f, num_parallel_calls=AUTOTUNE)
        ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE, seed=1234)

    # RGB to YCbCr
    if luma:
        ds = ds.map(rgb2ycbcr, num_parallel_calls=AUTOTUNE)

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Use buffered prefetching on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


def prepare2(ds, batch_size):
    # Resize and rescale all datasets
    ds = ds.map(resize_and_rescale2, num_parallel_calls=AUTOTUNE)

    # RGB to YCbCr
    ds = ds.map(rgb2ycbcr2, num_parallel_calls=AUTOTUNE)

    # Batch all datasets
    ds = ds.batch(batch_size)

    # Use buffered prefetching on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


# def test(data):
#     for x, y in data.take(1):
#         m = np.squeeze(x[0].numpy())
#         n = np.squeeze(y[0].numpy())
#
#         plt.figure()
#         plt.imshow(m)
#         plt.show()
#
#         plt.figure()
#         plt.imshow(n)
#         plt.show()
#     return m, n


def load_data(ds_trn, ds_val, ds_tst, sz_trn, batch_size, aug):
    # Train dataset
    if ds_trn == 'nyudv2':
        data_trn = load_nyudv2(sz_trn)
    elif ds_trn == 'diode':
        data_trn = load_diode(sz_trn)
    else:
        return
    data_trn = prepare(data_trn, batch_size=batch_size, shuffle=True, aug=aug)

    # Validation dataset
    if ds_val == 'middlebury':
        data_val = load_middlebury()
    elif ds_val == 'd_haze':
        data_val = load_dense_haze()
    elif ds_val == 'nh_haze':
        data_val = load_nh_haze()
    else:
        return
    data_val = prepare(data_val, batch_size=4)

    # Test dataset
    # if ds_tst == 'sots':
    #     data_tst1 = load_sots_indoor()
    #     data_tst1 = prepare(data_tst1, batch_size=50)
    #
    #     data_tst2 = load_sots_outdoor()
    #     data_tst2 = prepare(data_tst2, batch_size=50)

    # m, n = test(test_data)
    # return
    return data_trn, data_val


def load_test_data(data, batch_size):
    if data == 'mbury':
        ds = load_middlebury()
    elif data == 'sots_in':
        ds = load_sots_indoor()
    elif data == 'sots_out':
        ds = load_sots_outdoor()
    elif data == 'dense_haze':
        ds = load_dense_haze()
    elif data == 'nh_haze':
        ds = load_nh_haze()
    elif data == 'rain12':
        ds = load_rain12()
    ds = prepare(ds, batch_size=batch_size)
    return ds


def load_test_data2(data, batch_size):
    if data == 'colorlines':
        ds = load_colorlines()
    ds = prepare2(ds, batch_size)
    return ds

# load_data(trn_ds='diode', val_ds='middlebury', tst_ds='sots_out', batch_size=64)
