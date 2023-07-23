#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=ad2imgnet32sgd

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

python3 ../run_adam.py --dir=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --decoupled_wd --epochs=225  --swag --swag_start=176 --swag_lr=0.003 --save_freq=25 --use_test --lr_init=0.003 --wd=0.01 --eval_freq=1
