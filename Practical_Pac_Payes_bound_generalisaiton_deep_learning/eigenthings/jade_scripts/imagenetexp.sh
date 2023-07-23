#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=imgnet32swa

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

# Weight decay = 0
python3 run_swag.py --dir=./ckpts/32imagenet/PreResNet110_swag/ --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=225 --save_freq=25 --lr_init=0.1 --wd=5e-4 --seed=5123 --swag --use_test --swag_subspace=pca --swag_rank=20 --swag_start=161 --swag_lr=0.01 --use_test --use_test

