#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=KFAC110

#choose partition
#SBATCH --partition=devel

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=xingchen.wan@outlook.com

# run the application
module load python3/anaconda
source activate diegorubin

# KFAC
# Weight decay = 0
python3 run_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/PreResNet110/KFAC/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3 --wd 1e-4 --epochs 150 --save_freq=25