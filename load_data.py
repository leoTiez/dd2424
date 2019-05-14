#!/usr/bin/python2

"""load_data.py: Load and preprocess datasets"""

__author__ = "Adrian Chmielewski-Anders, Leo Zeitler & Bas Straathof"

from tensorflow.contrib.learn.python.learn.datasets.mnist import \
        extract_images, extract_labels
from tensorflow.python.keras.utils import to_categorical
import cPickle
import numpy as np
import random
from os import environ

DATA_DIR = './data'
if 'DATA_DIRECTORY' in environ:
    DATA_DIR = environ['DATA_DIRECTORY']


def to_greyscale(data, dtype):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


def preprocess_cifar_imgs_(data, dtype, use_grayscale=True):
    """Preprocesses the images in the cifar10 or cifar100 dataset

    Args:
        data (np.ndarray): data array

    Returns:
        data (np.ndarray): preprocessed data array
    """

    batch_size = data.shape[0]
    R = data[:, 0:1024] / 255.
    G = data[:, 1024:2048] / 255.
    B = data[:, 2048:] / 255.
    data = np.dstack((R, G, B)).reshape((batch_size, 32, 32, 3))

    if use_grayscale:
        data = to_greyscale(data, dtype)

    return data


def data_loader(dataset, file_name, dtype, use_grayscale=False,
        data_length=None, coarse=False):
    """Loads a dataset and preprocesses it

    Args:
        dataset   (str): identifier of the dataset to be used; for now,
                         this can be either one of 'mnist', 'cifar10' or
                         'cifar100'.
        file_name (str): filename of the data batch to be loaded
        dtype    (type): the datatype precision to be used
        coarse   (bool): used to speicify coarse labelling of CIFAR100

    Returns:
        data (np.ndarray): data matrix (D, N)
        labels (np.ndarray): labels matrix
    """
    data = None
    labels = None
    if dataset == "mnist":
        with open(DATA_DIR + '/mnist/' + 'mnist-' +
                file_name + '-images.gz', 'rb') as fo:
            data = np.asarray(extract_images(fo), dtype=dtype) / 255.

        with open(DATA_DIR + '/mnist/' + 'mnist-' + file_name +
                '-labels.gz', 'rb') as fo:
            labels = to_categorical(
                np.asarray(extract_labels(fo), dtype=dtype),
                num_classes=10
            )

    elif dataset == "cifar10":
        with open(str(DATA_DIR + '/cifar-10-batches-py/' + file_name), 'rb') as fo:
            data_dict = cPickle.load(fo)

        data = preprocess_cifar_imgs_(np.asarray(data_dict['data'], dtype=dtype),
                                      dtype=dtype,
                                      use_grayscale=use_grayscale
                                      )
        labels = to_categorical(np.asarray(data_dict['labels'], dtype=dtype),
                num_classes=10)

    elif dataset == "cifar100":
        with open(str(DATA_DIR + '/cifar-100-python/' + file_name), 'rb') as fo:
            data_dict = cPickle.load(fo)

        data = preprocess_cifar_imgs_(np.asarray(data_dict['data'], dtype=dtype),
                                      dtype=dtype,
                                      use_grayscale=use_grayscale
                                      )
        if bool(coarse):
            labels = to_categorical(np.asarray(data_dict['coarse_labels'], dtype=dtype),
                    num_classes=20)
        else:
            labels = to_categorical(np.asarray(data_dict['fine_labels'], dtype=dtype),
                    num_classes=100)

    if data_length is None:
        return data, labels
    else:
        data_zip = list(zip(data, labels))
        random.shuffle(data_zip)
        data, labels = zip(*data_zip)
        return np.asarray(data)[:data_length], np.asarray(labels)[:data_length]
