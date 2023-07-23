#!/bin/bash

source activate curvature

#SGDSWA
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --wd 1e-4 --epochs 300 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr 0.05 --swag_start 161 --use_test

#AdamX
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0.25 --epochs 300 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr 0.00025 --swag_start 161 --use_test --decoupled_wd

#AdamW
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0.25 --epochs 300 --save_freq=25 --eval_freq=5 --use_test --decoupled_wd

#SGD
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --wd 1e-4 --epochs 300 --save_freq=25 --eval_freq=5 --use_test

#Adam
python3 run_adam.py --dir /nfs/home/xingchenw/curvature/out --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-4 --epochs 300 --save_freq=25 --eval_freq=5 --use_test

