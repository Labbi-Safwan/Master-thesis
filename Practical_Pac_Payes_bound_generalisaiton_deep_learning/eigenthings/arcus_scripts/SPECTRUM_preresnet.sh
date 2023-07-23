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
python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFACW/KFACW-00150_train.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFACW/KFACW_wd2.5e-4/checkpoint-00150.pt --iters 50

# KFACWSWA

python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFACWSWA/KFACWSWA-00150_train.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFACWSWA/KFACW_wd2.5e-4/swag-00150.pt --iters 50 --swag


# AdamW
python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW-00300_train.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd0.25/checkpoint-00300.pt --iters 50


# AdamWSWA

python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamWSWA/AdamWSWA-00300-train.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamWSWA/AdamW_wd0.25/swag-00300.pt --iters 50 --swag

# SGD_noschedule
python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SGD_noschedule/SGD_noschedule-00300_train.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SGD_noschedule/SGD_wd5e-4/checkpoint-00300.pt --iters 50


# SGDSWA_noschedule

python3 spectrum.py --spectrum_path /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SGD_noschedule/SGDSWA_noschedule-00300-train.npz --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --ckpt /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/SGDSWA_noschedule/SGD_wd5e-4/swag-00300.pt --iters 50 --swag

