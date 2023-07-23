#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=PreResNet110

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=xingchen.wan@outlook.com

# run the application
source activate curvature

# SWATS-L2
# Weight decay = 0
python3 run_SWATS.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/SWATS/SWATS_wd0/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-5
python3 run_SWATS.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/SWATS/SWATS_wd1e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-5 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-5
python3 run_SWATS.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/SWATS/SWATS_wd5e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-5 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-4
python3 run_SWATS.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/SWATS/SWATS_wd1e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-4 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-4 - this is not needed as we already have data for wd = 5e-4
python3 run_SWATS.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/SWATS/SWATS_wd5e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-4 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-3
python3 run_SWATS.py --dir /nfs/home/xingchenw/curvature/out/VGG16BN/SWATS/SWATS_wd1e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-3 --epochs 300 --save_freq=5 --decoupled_wd
