#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=imgnettest

#choose partition
#SBATCH --partition=devel

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

# Weight decay = 0
python3 run_swag.py --dir=./ckpts/32imagenet/test/ --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16BN --epochs=225 --save_freq=50 --lr_init=0.05 --wd=5e-4 --seed=5123 --swag --swag_subspace=pca --swag_rank=20 --swag_start=161 --swag_lr=0.01

