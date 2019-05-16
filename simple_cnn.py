#!/usr/bin/python2

from __future__ import absolute_import, division, unicode_literals

"""simple_cnn.py: Implementation of a simple convolutional neural network
for image classification using Keras. Inspired by:
    https://www.tensorflow.org/alpha/tutorials/images/intro_to_cnns

Usage from CLI: $ python simple_cnn.py <dataset> <*number_of_runs>
Where <dataset> is one of 'mnist', 'cifar10' or 'cifar100' and
<number_of_runs> is an optional input parameter.

Part of a project for the 2019 DD2424 Deep Learning in Data Science course at
KTH Royal Institute of Technology"""

__author__ = "Adrian Chmielewski-Anders, Bas Straathof & Leo Zeitler"

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress warnings

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
tf.logging.set_verbosity(tf.logging.ERROR) # suppress warnings

from load_data import data_loader

# Constants
PRECISION_NP = np.float32


def cnn_model(train_images, train_labels, test_images, test_labels, input_shape,
        num_classes, num_epochs):
    """Simple CNN model

    Layers:
      - convolutional layer
      - max pooling layer
      - convolutional layer
      - max pooling layer
      - convolutional layer
      - fully-connected layer
      - fully-connected layer

    Args:
        train_images (np.ndarray): training images
        train_labels (np.ndarray): training labels
        test_images  (np.ndarray): test images
        test_labels  (np.ndarray): test labels
        input_shape       (tuple): shape of the input to the first layer
        num_classes         (int): number of classes // output shape
        num_epochs          (int): number of training epochs

    Returns:
        test_acc (float): the test accuracy
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=num_epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    return test_acc


def main(argv):
    """Main function to run the models

    Args:
        argv (list): CLI arguments
            - argv[1] (str): expected to be name of dataset to be used
            - argv[2] (str): expected to specify the number of runs
    """
    number_of_runs = 1
    if len(argv) > 2:
        number_of_runs = int(argv[2])

    dataset_name_  = argv[1]
    if dataset_name_.upper() == "MNIST":
        input_shape_ = (28, 28, 1)
        num_classes_ = 10
        num_epochs_  = 5

        train_images_, train_labels_ = data_loader("mnist", "train", dtype=np.float32)
        test_images_, test_labels_ = data_loader("mnist", "test", dtype=np.float32)
        train_labels_ = np.argmax(train_labels_, axis=1) # one-hot-decoding
        test_labels_  = np.argmax(test_labels_, axis=1)  # one-hot-decoding

    elif dataset_name_.upper() == "CIFAR10":
        grayscale_   = False
        input_shape_ = (32, 32, 3)
        num_classes_ = 10
        num_epochs_  = 10

        train_images_ = np.empty((0, 32, 32, 3))
        train_labels_ = np.empty((0, 10))

        for i in range(1, 6):
            train_images_batch_, train_labels_batch_ = data_loader(
                "cifar10",
                "data_batch_" + str(i),
                dtype=np.float32,
                use_grayscale=grayscale_)

            train_images_ = np.concatenate((train_images_, train_images_batch_))
            train_labels_ = np.concatenate((train_labels_, train_labels_batch_))

            test_images_, test_labels_ = data_loader( "cifar10", "test_batch",
                    dtype=np.float32, use_grayscale=grayscale_)

        train_labels_ = np.argmax(train_labels_, axis=1) # one-hot-decoding
        test_labels_  = np.argmax(test_labels_, axis=1)  # one-hot-decoding

    elif dataset_name_.upper() == "CIFAR100":
        grayscale_   = False
        input_shape_ = (32, 32, 3)
        num_classes_ = 100
        num_epochs_  = 12

        train_images_, train_labels_ = data_loader("cifar100", "train",
                dtype=np.float32, use_grayscale=grayscale_)

        test_images_, test_labels_ = data_loader("cifar100", "test",
                dtype=np.float32, use_grayscale=grayscale_)

        train_labels_ = np.argmax(train_labels_, axis=1) # one-hot-decoding
        test_labels_  = np.argmax(test_labels_, axis=1)  # one-hot-decoding

    else:
        raise Exception("dataset has to be one of 'mnist', 'cifar10' or 'cifar100'.")

    test_accs = []
    for i in range(number_of_runs):
        test_acc = cnn_model(train_images=train_images_, train_labels=train_labels_,
                test_images=test_images_, test_labels=test_labels_,
                input_shape=input_shape_, num_classes=num_classes_,
                num_epochs=num_epochs_)
        test_accs.append(test_acc)

    print "Test mean acc:" + str(np.mean(np.asarray(test_accs))) + \
            " for %s runs." % number_of_runs
    print "Test stdev acc:" + str(np.std(np.asarray(test_accs))) + \
            " for %s runs." % number_of_runs


if __name__ == '__main__':
    main(sys.argv)

