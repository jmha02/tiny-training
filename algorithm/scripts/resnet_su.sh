# sparse update

python train_cls.py configs/cifar10.yaml --run_dir runs/cifar10/resnet18/su \
    --net_name resnet18 --base_lr 0.1 --optimizer_name sgd_nomom \
    --enable_backward_config 1 --n_bias_update 8 --manual_weight_idx 12-15-17-19 \
    --weight_update_ratio 1-0.5-0.25-0.125 --n_epochs 50

python train_cls.py configs/cifar10.yaml --run_dir runs/cifar10/resnet18/full \
    --net_name resnet18 --base_lr 0.1 --optimizer_name sgd_nomom --n_epochs 50