# sparse update - increased layers to train more parameters
python train_cls.py configs/cifar10.yaml --run_dir runs/cifar10/resnet18/su \
    --net_name resnet18 --base_lr 0.1 --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 10 --n_weight_update 10 --n_epochs 10
# update full network
python train_cls.py configs/cifar10.yaml --run_dir runs/cifar10/resnet18/full \
    --net_name resnet18 --base_lr 0.1 --optimizer_name sgd_scale_nomom --n_epochs 10