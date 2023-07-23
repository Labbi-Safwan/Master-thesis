#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=100:59:59
#SBATCH --gres=gpu:1
# set name of job
#SBATCH --job-name=KFACW

#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature


# KFAC
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/VGG16BN/KFAC/KFAC_wd0./ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.5 --wd 0. --epochs=150 --save_freq=25 --eval_freq=5

