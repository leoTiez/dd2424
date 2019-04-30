#!/usr/bin/env bash

OS_NAME ="$(uname -s)"

if ["$OS_NAME" == "Linux"]; then
    sudo apt-get update && sudo apt-get install curl
fi

DATA_DIRECTORY="DATA"
if [! -d "$DATA_DIRECTORY"]; then
  mkdir data
fi

CIFAR_10_DIRECTORY="cifar-10-batches-py"
if [! -d "$CIFAR_10_DIRECTORY"]; then
    curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-batches-py.tar.gz
    tar -xvzf data/cifar-10-batches-py.tar.gz -C data/
    rm -rf data/cifar-10-batches-py.tar.gz
fi

CIFAR_100_DIRECTORY="cifar-100-batches-py"
if [! -d "$CIFAR_100_DIRECTORY"]; then
    curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o data/cifar-100-batches-py.tar.gz
    tar -xzvf data/cifar-100-batches-py.tar.gz -C data/
    rm -rf data/cifar-10-batches-py.tar.gz
fi

MNIST_DIRECTORY="mnist"
if [! -d "$MNIST_DIRECTORY"]; then
    mkdir mnist
    curl yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o data/mnist-train-images.tar.gz
    tar -xzvf data/mnist-train-images.tar.gz -C data/

    curl yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o data/mnist-train-labels.tar.gz
    tar -xzvf data/mnist-train-labels.tar.gz -C data/

    curl yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o data/mnist-test-images.tar.gz
    tar -xzvf data/mnist-test-images.tar.gz -C data/

    curl yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -o data/mnist-test-labels.tar.gz
    tar -xzvf data/mnist-test-labels.tar.gz -C data/
fi
