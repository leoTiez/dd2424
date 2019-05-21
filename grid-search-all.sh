#!/bin/bash

annealing_rate_list=(1 0.1 0.01)
depth_list=(0 3 6)
batch_size=1
learning_data_set_size=2000
testing_data_set_size=500
epochs=6
device=gpu

for annealing_rate in "${annealing_rate_list}";
do
    for depth in "${depth_list}";
    do
        python main.py "${device}" mnist -dl "${depth}" -dt "${depth}" -f "${annealing_rate}" -b "${batch_size}" \
            -lls "${learning_data_set_size}" -lts "${testing_data_set_size}" -nf 64 -e "${epochs}"
        python main.py "${device}" cifar10 -dl "${depth}" -dt "${depth}" -f "${annealing_rate}" -b "${batch_size}" \
            -lls "${learning_data_set_size}" -lts "${testing_data_set_size}" -nf 96 -e "${epochs}"
        python main.py "${device}" cifar100 -dl "${depth}" -dt "${depth}" -f "${annealing_rate}" -b "${batch_size}" \
            -lls "${learning_data_set_size}" -lts "${testing_data_set_size}" -nf 96 -e "${epochs}"
    done
done
