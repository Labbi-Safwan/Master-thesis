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

# KFACW
python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFACW/KFACW-00150_train_nowd.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd0/checkpoint-00150.pt --iters 50


# AdamW
python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW-00300_train_nowd.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/Adam/Adam_wd0/checkpoint-00300.pt --iters 50

# SGD_noschedule
#python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SGD_noschedule/SGD_noschedule-00300_train_nowd.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SGD_noschedule/SGD_wd0/checkpoint-00300.pt --iters 50

