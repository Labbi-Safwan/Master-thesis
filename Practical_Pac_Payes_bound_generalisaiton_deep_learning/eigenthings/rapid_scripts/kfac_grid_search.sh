#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

source activate curvature

# KFAC
# Learning rate 0.1
python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.1d0.5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.5

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.1d0.1/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.1

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.1d0.05/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.1 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.05

# Leanring rate 0.05

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.051d0.1/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.05 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.1

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.05d0.05/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.05 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.05

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.05d0.01/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.05 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.01

# Leanring rate 0.01

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.01d0.05/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.01 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.05

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.01d0.01/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.01 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.01

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.01d0.005/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.01 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.005

# Leanring rate 0.005

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.005d0.01/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.01

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.005d0.005/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.005

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.005d0.001/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.005 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.001

# Leanring rate 0.001

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.001d0.005/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.001 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.005

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.001d0.001/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.001 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.001

python3 run_KFAC.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/GridSearch/KFAC_lr0.001d0.0005/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.001 --wd 5e-4 --epochs=75 --save_freq=75 --damping 0.0005
