from sys import argv

dataset = argv[1]

cnn = lambda x: x == '0'
rcnn = lambda x: x == '3' or x == '6'

with open('accuracies-{}.txt'.format(dataset)) as f:
    all_configs = f.readlines()

all_configs = [x.strip().split(' ') for x in all_configs]

acc_config_map = {}

nn_type = cnn if argv[2] == 'cnn' else rcnn

for config in all_configs:
    depth = config[2][:-1]
    if nn_type(depth):
        acc_config_map[float(config[-1])] = [
            '-d', depth,
            '-f', config[5][:-1],
            '-b', config[7][:-1]
        ]

best_acc = max(acc_config_map.keys())

print ' '.join(acc_config_map[best_acc])
