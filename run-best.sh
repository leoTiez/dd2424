datasets=(mnist cifar10 cifar100)

device=${1:-gpu}

for ds in "${datasets[@]}" ; do
    argc=`python train_best.py $ds cnn`
    echo $argc
    python main.py gpu $ds $argc

    argr=`python train_best.py $ds rcnn`
    echo $argr
    python main.py gpu $ds $argr
done

