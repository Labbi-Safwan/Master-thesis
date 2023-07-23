#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
# set name of job
#SBATCH --job-name=Adam
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Volta'
#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature

# AdamW

# Weight decay = 1
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/VGG16BN/AdamW/AdamW_wd1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/VGG16BN/AdamW/AdamW_wd5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5 --epochs 300 --save_freq=5 --decoupled_wd

# AdamWSWA
# Weight decay = 1
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.00025

# Weight decay = 5
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.00025

