#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Volta'
# set name of job
#SBATCH --job-name=Adam

#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature

# Weight decay = 5e-1
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25

