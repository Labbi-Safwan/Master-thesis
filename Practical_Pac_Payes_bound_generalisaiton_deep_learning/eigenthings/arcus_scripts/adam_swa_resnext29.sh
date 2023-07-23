#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Volta'
# set name of job
#SBATCH --job-name=Adamw

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
# module load python3/anaconda
source activate curvature

# Adam-L2
# Weight decay = 0
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamSWA_noschedule/Adam_wd0/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 0 --epochs 300 --eval_freq=5 --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.0005

# Weight decay = 1e-5
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamSWA_noschedule/Adam_wd1e-5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1e-5 --epochs 300 --eval_freq=5 --swag --no_covariance --swag_start=161 --swag_lr=0.0005 --save_freq=25

# Weight decay = 5e-5
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamSWA_noschedule/Adam_wd5e-5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 5e-5 --epochs 300 --eval_freq=5 --swag --no_covariance --swag_start=161 --swag_lr=0.0005 --save_freq=25

# Weight decay = 1e-4
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamSWA_noschedule/Adam_wd1e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1e-4 --epochs 300 --eval_freq=5 --swag --no_covariance --swag_start=161 --swag_lr=0.0005 --save_freq=25

# Weight decay = 5e-4
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamSWA_noschedule/Adam_wd5e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 5e-4 --epochs 300 --eval_freq=5 --swag --no_covariance --swag_start=161 --swag_lr=0.0005 --save_freq=25

# Weight decay = 1e-3
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamSWA_noschedule/Adam_wd1e-3/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1e-3 --epochs 300 --eval_freq=5 --swag --no_covariance --swag_start=161 --swag_lr=0.0005 --save_freq=25


