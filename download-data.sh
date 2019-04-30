#!/usr/bin/env bash

unameOut="$(uname -s)"

case "${unameOut}" in
    Linux*)     sudo apt-get update && sudo apt-get install curl;;
    CYGWIN*)    sudo apt-get update && sudo apt-get install curl;;
    MINGW*)     sudo apt-get update && sudo apt-get install curl;;
esac

rm -rf data/
mkdir data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o data/cifar-10-batches-py.tar.gz
tar -xvzf data/cifar-10-batches-py.tar.gz -C data/
rm -rf data/cifar-10-batches-py.tar.gz

curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o data/cifar-100-batches-py.tar.gz
tar -xzvf data/cifar-100-batches-py.tar.gz -C data/
rm -rf data/cifar-10-batches-py.tar.gz
