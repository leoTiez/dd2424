from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import cPickle
import numpy as np


def load_mnist(file_name, dtype, file_path="./data/mnist/"):
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


def load_cifar(file_name, dtype, file_path='./data/cifar-10-batches-py/'):
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


