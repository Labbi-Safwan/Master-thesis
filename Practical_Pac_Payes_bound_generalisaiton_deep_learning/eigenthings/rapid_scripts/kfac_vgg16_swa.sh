#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

source activate curvature

# KFAC-SWA
python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA2/KFAC_wd0/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 0 --epochs=150 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA2/KFAC_wd1e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 1e-5 --epochs=150 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA2/KFAC_wd5e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 5e-5 --epochs=150 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA2/KFAC_wd1e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 1e-4 --epochs=150 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA2/KFAC_wd5e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 5e-4 --epochs=150 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr=0.05 --swag_start=81

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA2/KFAC_wd1e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 1e-3 --epochs=150 --save_freq=25 --eval_freq=5 --swag --no_covariance --swag_lr=0.05 --swag_start=81



