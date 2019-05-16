#!/bin/bash

# Ensure python points to python2.7
# Run the simple CNN model 10 times on each dataset
python simple_cnn.py mnist 10
python simple_cnn.py cifar10 10
python simple_cnn.py cifar100 10
