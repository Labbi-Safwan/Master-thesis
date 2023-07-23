#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Volta'
# set name of job
#SBATCH --job-name=AdamNeXt

#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature

# Weight decay = 5e-4
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/Adam/Adam_wd5e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 5e-4 --epochs 300 --eval_freq=5 --save_freq=25

# Weight decay = 1e-3
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/Adam/Adam_wd1e-3/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1e-3 --epochs 300 --eval_freq=5 --save_freq=25
