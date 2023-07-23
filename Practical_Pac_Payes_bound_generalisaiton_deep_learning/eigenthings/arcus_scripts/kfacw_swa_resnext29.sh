#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59
# set name of job
#SBATCH --job-name=Adamw

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
# module load python3/anaconda
source activate curvature

# Weight decay = 5e-3
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/KFACWSWA/KFACW_wd1e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.1 --damping 0.3  --wd 1e-4 --epochs 150 --eval_freq=5 --decoupled_wd --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.05

# Weight decay = 1e-2
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/KFACWSWA/KFACW_wd2.5e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.1 --damping 0.3  --wd 2.5e-4 --epochs 150 --eval_freq=5 --decoupled_wd --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.05

python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/KFACWSWA/KFACW_wd5e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.1 --damping 0.3  --wd 5e-4 --epochs 150 --eval_freq=5 --decoupled_wd --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.05

# Weight decay = 5e-2
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/KFACWSWA/KFACW_wd1e-3/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.1 --damping 0.3  --wd 1e-3 --epochs 150 --eval_freq=5 --decoupled_wd --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.05

# Weight decay = 1e-1
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/KFACWSWA/KFACW_wd5e-3/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.1 --damping 0.3  --wd 5e-3 --epochs 150 --eval_freq=5 --decoupled_wd --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.05

# Weight decay = 5e-1
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/ResNeXt29CIFAR/KFACWSWA/KFACW_wd1e-2/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.1 --damping 0.3  --wd 1e-2 --epochs 150 --eval_freq=5 --decoupled_wd --save_freq=25 --swag --no_covariance --swag_start=161 --swag_lr=0.05

