#!/bin/bash

# ensure python points to python2.7
python main.py gpu mnist grid-search
python main.py gpu cifar10 grid-search
python main.py gpu cifar100 grid-search
