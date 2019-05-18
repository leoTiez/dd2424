#!/bin/bash

lname="train-3-test-6"


python main.py gpu mnist -dl 3 -dt 6 -f 0.01 -b 1 > ${lname}-mnist.log \
    2> ${lname}-mnist.err.log

python main.py gpu cifar10 -dl 3 -dt 6 -f 0.1 -b 1 > ${lname}-cifar10.log \
    2> ${lname}-cifar10.err.log

python main.py gpu cifar100 -dl 3 -dt 6 -f 1 -b 1 > ${lname}-cifar100.log \
    2> ${lname}-cifar100.err.log