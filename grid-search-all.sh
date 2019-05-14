#!/bin/bash

# ensure python points to python2.7
python main gpu mnist grid-search
python main gpu cifar10 grid-search
python main gpu cifar100 grid-search
