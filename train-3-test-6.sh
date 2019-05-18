#!/bin/bash

lname="train-3-test-6"

python main.py gpu mnist -dl 3 -dt 6 -f 0.1 -b 1 > ${lname}.log \
    2> ${lname}.err.log
