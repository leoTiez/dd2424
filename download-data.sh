#!/usr/bin/env bash

OS_NAME="$(uname -s)"

if [[ "$OS_NAME" == "Linux" ]]; then
    sudo apt-get update && sudo apt-get install curl
fi

DATA_DIRECTORY="data"
if [[ ! -d "$DATA_DIRECTORY" ]]; then
  mkdir data
fi

CIFAR_10_DIRECTORY="data/cifar-10-batches-py"
if [[ ! -d "$CIFAR_10_DIRECTORY" ]]; then
    curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-batches-py.tar.gz
    tar -xvf data/cifar-10-batches-py.tar.gz -C data/
    rm -rf data/cifar-10-batches-py.tar.gz
fi

CIFAR_100_DIRECTORY="data/cifar-100-python"
if [[ ! -d "$CIFAR_100_DIRECTORY" ]]; then
    curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o data/cifar-100-batches-py.tar.gz
    tar -xvf data/cifar-100-batches-py.tar.gz -C data/
    rm -rf data/cifar-100-batches-py.tar.gz
fi

MNIST_DIRECTORY="data/mnist"
if [[ ! -d "$MNIST_DIRECTORY" ]]; then
    mkdir data/mnist
    curl yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o data/mnist/mnist-train-images.gz
    curl yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o data/mnist/mnist-train-labels.gz
    curl yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o data/mnist/mnist-test-images.gz
    curl yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -o data/mnist/mnist-test-labels.gz
fi
