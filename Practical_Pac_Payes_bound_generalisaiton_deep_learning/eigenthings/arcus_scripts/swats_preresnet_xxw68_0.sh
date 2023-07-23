#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
# set name of job
#SBATCH --job-name=SWATS0

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
# module load python3/anaconda
source activate curvature

# SWATS-L2
# Weight decay = 0
python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-5
python3 run_SWATS.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SWATS/SWATS_wd0.001/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 0.001 --epochs 300 --save_freq=5 --decoupled_wd

