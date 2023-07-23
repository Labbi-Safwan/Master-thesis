#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=adamswa

#choose partition
#SBATCH --partition=small

#choose time limit
#SBATCH --time=23:59:59

# set number of GPUs
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Volta'

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
source activate pytorch
# Weight decay = 0
python3 run_adam.py --dir=/data/parg/chri3937/data/Adam/ --no_covariance --dataset=ImageNet32 --data_path=/data/parg/chri3937/data/ --model=VGG16BN --epochs=200 --save_freq=50 --lr_init=0.0005 --wd=0.5 --eval_freq=5 --decoupled_wd --seed=5123 --swag --swag_subspace=pca --swag_rank=20 --swag_start=111 --swag_lr=0.00025

