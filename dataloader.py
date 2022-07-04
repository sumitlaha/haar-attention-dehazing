from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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
    # Standardize non-hazy image
    image = tf.cast(image, tf.float32)
    image /= 255.0
    # Standardize image depth
    depth = tf.cast(depth, tf.float32)
    depth = standardize(depth, axes=[0, 1])
    if invert_depth:
        depth = 1 - depth
    im_depth = tf.expand_dims(depth, axis=-1)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(SEED, num=2)
    # Initialize atmospheric light with random value between 0.7 and 1.0
    atm_l = tf.random.stateless_uniform(shape=[], seed=new_seed[0, :], minval=0.7, maxval=1.0, dtype=tf.float32)
    # Initialize scattering coefficient with random value between 0.5 and 1.3
    beta = tf.random.stateless_uniform(shape=[mul_beta], seed=new_seed[1, :], minval=0.5, maxval=1.3, dtype=tf.float32)

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


def resize_and_rescale(im_hzy, im_clr):
    imgs = tf.concat([im_hzy, im_clr], axis=-1)
    imgs = tf.keras.preprocessing.image.smart_resize(imgs, size=[IMG_HT, IMG_WD])
    imgs = tf.clip_by_value(imgs, 0, 1)
    im_hzy = imgs[..., :3]
    im_clr = imgs[..., 3:]
    return im_hzy, im_clr


def prepare(ds, batch_size, shuffle=False):
    # Resize and rescale all datasets
    ds = ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    # Shuffle data
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE, seed=1234)
    # Batch all datasets
    ds = ds.batch(batch_size)
    # Use buffered prefetching on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


def load_data(ds_train, ds_val, ds_test, sz_trn, batch_size=1):
    # Train dataset
    if ds_train == 'nyudv2':
        data_train = load_nyudv2(sz_trn)
    else:
        return
    data_train = prepare(data_train, batch_size=batch_size, shuffle=True)

    # Validation dataset
    if ds_val == 'middlebury':
        data_val = load_middlebury()
    else:
        return
    data_val = prepare(data_val, batch_size=1)

    # Test dataset
    if ds_test == 'd_haze':
        data_test = load_dense_haze()
    elif ds_test == 'nh_haze':
        data_test = load_nh_haze()
    else:
        return
    data_test = prepare(data_test, batch_size=1)
    return data_train, data_val, data_test
