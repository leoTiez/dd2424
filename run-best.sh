datasets=(mnist cifar10 cifar100)

for ds in "${datasets[@]}" ; do
    args=`python train_best.py $ds`
    echo $args
    python main.py gpu $ds $args
done

