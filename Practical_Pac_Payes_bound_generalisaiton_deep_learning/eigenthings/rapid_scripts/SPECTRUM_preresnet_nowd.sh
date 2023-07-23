#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1 --constraint='gpu_gen:Volta'
# set name of job
#SBATCH --job-name=Spectrum

#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature


# SGD_noschedule
python3 spectrum.py --spectrum_path  /nfs/home/xingchenw/curvature/out/PreResNet110/SGD_noschedule/SGD_noschedule-00300_train_nowd.npz --dataset CIFAR100 --data_path  /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --ckpt  /nfs/home/xingchenw/curvature/out/PreResNet110/SGD_noschedule/SGD_wd0/checkpoint-00300.pt --iters 50

