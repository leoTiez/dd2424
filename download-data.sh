#!/usr/bin/env bash

OS_NAME="$(uname -s)"

if [[ "$OS_NAME" == "Linux" ]]; then
    sudo apt-get update && sudo apt-get install curl
fi

DATA_DIRECTORY="data"
if [[ ! -d "$DATA_DIRECTORY" ]]; then
  mkdir $DATA_DIRECTORY
fi

CIFAR_10_DIRECTORY="${DATA_DIRECTORY}/cifar-10-batches-py"
if [[ ! -d "$CIFAR_10_DIRECTORY" ]]; then
    curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-batches-py.tar.gz
    tar -xvf data/cifar-10-batches-py.tar.gz -C data/
    rm -rf data/cifar-10-batches-py.tar.gz
fi

CIFAR_100_DIRECTORY="${DATA_DIRECTORY}/cifar-100-python"
if [[ ! -d "$CIFAR_100_DIRECTORY" ]]; then
    curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o data/cifar-100-batches-py.tar.gz
    tar -xvf $DATA_DIRECTORY/cifar-100-batches-py.tar.gz -C $DATA_DIRECTORY
    rm -rf $DATA_DIRECTORY/cifar-100-batches-py.tar.gz
fi

MNIST_DIRECTORY="${DATA_DIRECTORY}/mnist"
if [[ ! -d "$MNIST_DIRECTORY" ]]; then
    mkdir $MNIST_DIRECTORY
    curl yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o \
        $MNIST_DIRECTORY/mnist-train-images.gz
    curl yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o \
        $MNIST_DIRECTORY/mnist-train-labels.gz
    curl yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o \
        $MNIST_DIRECTORY/mnist-test-images.gz
    curl yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -o \
        $MNIST_DIRECTORY/mnist-test-labels.gz
fi

TINY_NET_DIRECTORY="${DATA_DIRECTORY}/tinynet"
if [[ ! -d "$TINY_NET_DIRECTORY" ]]; then
    mkdir $TINY_NET_DIRECTORY
    curl http://cs231n.stanford.edu/tiny-imagenet-200.zip -o \
        $TINY_NET_DIRECTORY/raw.zip
    tar -xf $TINY_NET_DIRECTORY/raw.zip -C $TINY_NET_DIRECTORY --strip-components=1
fi
export DATA_DIRECTORY
