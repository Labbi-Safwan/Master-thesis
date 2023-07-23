#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

source activate curvature

# AdamWSWA
# Weight decay = 5e-3
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-3 --epochs 300 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_start=161 --swag_lr=0.00025

# Weight decay = 1e-2
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1e-2/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-2 --epochs 300 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_start=161 --swag_lr=0.00025

# Weight decay = 5e-2
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-2/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-2 --epochs 300 --save_freq=25  --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_start=161 --swag_lr=0.00025

# Weight decay = 0.1
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1e-1/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0.1 --epochs 300 --save_freq=25  --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_start=161 --swag_lr=0.00025

# Weight decay = 0.5
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0.5 --epochs 300 --save_freq=25  --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_start=161 --swag_lr=0.00025

