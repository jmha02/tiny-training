# sparse update

python train_cls.py configs/cifar10.yaml --run_dir runs/cifar10/resnet18/su_new \
    --net_name resnet18 --base_lr 0.1 --optimizer_name sgd_nomom \
    --enable_backward_config 1 --n_bias_update 8 --manual_weight_idx 12-13-14-15-16-17-18-19 \
    --weight_update_ratio 0.125-0.125-0.25-0.25-0.5-0.5-1-1 --n_epochs 50

python train_cls.py configs/cifar10.yaml --run_dir runs/cifar10/resnet18/full_new \
    --net_name resnet18 --base_lr 0.1 --optimizer_name sgd_nomom --n_epochs 50