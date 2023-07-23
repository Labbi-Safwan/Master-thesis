#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

source activate curvature

# KFAC
python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd1e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 1e-5 --epochs=150 --save_freq=25 --eval_freq=5

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd5e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 5e-5 --epochs=150 --save_freq=25 --eval_freq=5

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd1e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 1e-4 --epochs=150 --save_freq=25 --eval_freq=5

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd5e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 5e-4 --epochs=150 --save_freq=25 --eval_freq=5

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd1e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 1e-3 --epochs=150 --save_freq=25 --eval_freq=5

# KFACw
python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACW/KFACW_wd1e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 1e-3 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACW/KFACW_wd5e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 5e-3 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACW/KFACW_wd1e-2/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 1e-2 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACW/KFACW_wd5e-2/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 5e-2 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFACW/KFACW_wd1e-1/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --damping 0.05 --wd 1e-1 --epochs=150 --save_freq=25 --eval_freq=5 --decoupled_wd



