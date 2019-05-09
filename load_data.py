#!/usr/bin/python2

"""load_data.py: Load and preprocess datasets"""

__author__ = "Adrian Chmielewski-Anders, Leo Zeitler & Bas Straathof"

from tensorflow.contrib.learn.python.learn.datasets.mnist import \
        extract_images, extract_labels
from tensorflow.python.keras.utils import to_categorical
import cPickle
import numpy as np
from os import environ

DATA_DIR = './data'
if 'DATA_DIRECTORY' in environ:
    DATA_DIR = environ['DATA_DIRECTORY']


def preprocess_cifar_imgs_(data):
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

    return data


def data_loader(dataset, file_name, dtype):
    """Loads a dataset and preprocesses it

    Args:
        dataset (str): identifier of the dataset to be used; for now,
                       this can be either one of 'mnist', 'cifar10' or
                       'cifar100'.
        file_name (str): filename of the data batch to be loaded
        dtype (type): the datatype precision to be used

    Returns:
        data (np.ndarray): data matrix (D, N)
        labels (np.ndarray): labels matrix
    """
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

        data = preprocess_cifar_imgs_(np.asarray(data_dict['data'], dtype=dtype))
        labels = to_categorical(np.asarray(data_dict['labels'], dtype=dtype),
                num_classes=10)

    elif dataset == "cifar100":
        with open(str(DATA_DIR + '/cifar-100-python/' + file_name), 'rb') as fo:
            data_dict = cPickle.load(fo)

        data = preprocess_cifar_imgs_(np.asarray(data_dict['data'], dtype=dtype))
        labels = to_categorical(np.asarray(data_dict['fine_labels'], dtype=dtype),
                num_classes=100)

    return data, labels

