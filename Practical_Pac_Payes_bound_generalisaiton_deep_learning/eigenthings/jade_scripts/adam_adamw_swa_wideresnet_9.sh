#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=9WideResNet28x10

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin


# AdamW

# Weight decay = 5e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/WideResNet228x10/AdamW/wd25e-5_lr_1e-3/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model WideResNet28x10 --lr_init 0.001 --wd 0.25 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.0005

# Weight decay = 5e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/WideResNet228x10/AdamW/wd5e-4_lr_1e-3/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model WideResNet28x10 --lr_init 0.001 --wd 0.5 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.0005

