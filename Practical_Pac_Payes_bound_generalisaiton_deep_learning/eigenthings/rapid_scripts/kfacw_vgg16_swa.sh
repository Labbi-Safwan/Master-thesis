#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

source activate curvature

# KFACw-SWA
python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACWSWA2/KFACW_wd1e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 1e-4 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACWSWA2/KFACW_wd5e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 5e-4 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACWSWA2/KFACW_wd1e-2/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 1e-3 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACWSWA2/KFACW_wd5e-2/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 5e-3 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACWSWA2/KFACW_wd1e-1/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 1e-2 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd --swag --no_covariance --swag_lr=0.05 --swag_start=81



