from sys import argv

dataset = argv[1]

with open('accuracies-{}.txt'.format(dataset)) as f:
    all_configs = f.readlines()

all_configs = [x.strip().split(' ') for x in all_configs]

acc_config_map = {}

for config in all_configs:
    acc_config_map[float(config[-1])] = [
        '-d', config[2][:-1],
        '-f', config[5][:-1],
        '-b', config[7][:-1]
    ]

best_acc = max(acc_config_map.keys())

print ' '.join(acc_config_map[best_acc])
