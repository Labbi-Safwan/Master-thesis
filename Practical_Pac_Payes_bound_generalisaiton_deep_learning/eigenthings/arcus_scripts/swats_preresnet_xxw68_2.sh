#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
# set name of job
#SBATCH --job-name=SWATS2

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
# module load python3/anaconda
source activate curvature

# SWATS-L2
# Weight decay = 0
#python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-5
#python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0.001/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0.001 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-5
#python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0.005/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0.005 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-4
#python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0.1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0.1 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-4 - this is not needed as we already have data for wd = 5e-4
python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0.25/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0.25 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-3
python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0.5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0.5 --epochs 300 --save_freq=5 --decoupled_wd
