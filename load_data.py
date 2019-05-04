from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, \
    extract_labels
import cPickle
import numpy as np
from os import environ, path, listdir
import tensorflow as tf
import csv

DATA_DIR = './data'
if 'DATA_DIRECTORY' in environ:
    DATA_DIR = environ['DATA_DIRECTORY']


def load_mnist(file_name, dtype, file_path=DATA_DIR + '/mnist/'):
    data_dict = {}
    with open(file_path + 'mnist-' + file_name + '-images.gz', 'rb') as fo:
        data_dict['data'] = np.asarray(extract_images(fo), dtype=dtype)
    with open(file_path + 'mnist-' + file_name + '-labels.gz', 'rb') as fo:
        data_dict['labels'] = np.asarray(extract_labels(fo), dtype=dtype)

    return data_dict


def preprocess_mnist_data(mnist_data, dtype):
    mean = np.mean(mnist_data, axis=0)
    std = np.std(mnist_data, axis=0)

    return (mnist_data - mean) / (std + np.finfo(dtype).eps)


def load_cifar(file_name, dtype, file_path=DATA_DIR + '/cifar-10-batches-py/'):
    """
    Loads the cifar data file
    :param file_name: name of the file
    :param file_path: path to the cifar data file
    :return: The cifar data as a dictionary
    """
    with open(str(file_path + file_name), 'rb') as fo:
        data_dict = cPickle.load(fo)
    data_dict['data'] = np.asarray(data_dict['data'], dtype=dtype)
    data_dict['labels'] = np.asarray(data_dict['labels'], dtype=dtype)
    return data_dict


def preprocess_cifar_data(cifar_data, dtype):
    mean = np.mean(cifar_data, axis=0)
    std = np.std(cifar_data, axis=0)
    batch_size = cifar_data.shape[0]

    cifar_data = (cifar_data - mean) / (std - np.finfo(dtype).eps)

    R = cifar_data[:, 0:1024]
    G = cifar_data[:, 1024:2048]
    B = cifar_data[:, 2048:]

    cifar_data = np.dstack((R, G, B)).reshape((batch_size, 32, 32, 3))

    return cifar_data


def _tinynet_parse_ims(filename, label):
    image_string = tf.read_file(filename)
    im = tf.image.decode_jpeg(image_string, channels=3)
    im = tf.dtypes.cast(im, tf.float32)
    im /= 255.0
    return im, label


def _tinynet_build_id_map(file_path):
    ids = sorted(open(path.join(file_path, 'wnids.txt')).read().splitlines())
    return {ids[i]: i for i in range(len(ids))}


def _tinynet_build_val_map(val_file, id_map):
    val_map = {}
    with open(val_file) as annotations:
        reader = csv.reader(annotations, delimiter='\t')
        for row in reader:
            # so will be filename => id
            val_map[row[0]] = id_map[row[1]]
    return val_map


TINYNET_DIR = '/tinynet/'


def load_tinynet_train(dtype=None, file_path=DATA_DIR + TINYNET_DIR):
    id_map = _tinynet_build_id_map(file_path)
    im_locs = []
    im_labels = []

    for anId in id_map:
        im_dir = path.join(file_path, 'train', anId, 'images')
        # all the file names
        im_files = listdir(im_dir)
        im_locs += [path.join(im_dir, f) for f in im_files]
        im_labels += [id_map[anId]] * len(im_files)

    fnames = tf.constant(im_locs)
    labels = tf.constant(im_labels)

    dataset = tf.data.Dataset.from_tensor_slices((fnames, labels))
    return dataset.map(_tinynet_parse_ims)


def load_tinynet_test(dtype=None, file_path=DATA_DIR + TINYNET_DIR):
    id_map = _tinynet_build_id_map(file_path)
    val_map = _tinynet_build_val_map(
        path.join(file_path, 'val/val_annotations.txt'), id_map
    )

    fpaths = []
    ids = []
    for f in val_map:
        fpaths.append(path.join(file_path, 'val/images', f))
        ids.append(val_map[f])

    vpaths = tf.constant(fpaths)
    vlabels = tf.constant(ids)
    dataset = tf.data.Dataset.from_tensor_slices((vpaths, vlabels))

    return dataset.map(_tinynet_parse_ims)
