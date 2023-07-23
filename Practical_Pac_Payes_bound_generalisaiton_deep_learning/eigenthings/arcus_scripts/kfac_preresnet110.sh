#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

#SBATCH --gres=gpu:1
# set name of job
#SBATCH --job-name=KFAC

#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature

# KFAC

# Weight decay = 0
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd0/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3 --wd 0 --epochs 150 --eval_freq=5 --save_freq=25

# Weight decay = 1e-5
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd1e-5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3  --wd 1e-5 --epochs 150 --eval_freq=5 --save_freq=25

# Weight decay = 5e-5
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd5e-5/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3  --wd 5e-5 --epochs 150 --eval_freq=5 --save_freq=25

# Weight decay = 1e-4
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd1e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3  --wd 1e-4 --epochs 150 --eval_freq=5 --save_freq=25

# Weight decay = 5e-4
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd5e-4/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3  --wd 5e-4 --epochs 150 --eval_freq=5 --save_freq=25

# Weight decay = 1e-3
python3 run_KFAC.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/KFAC/KFAC_wd1e-3/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.1 --damping 0.3  --wd 1e-3 --epochs 150 --eval_freq=5 --save_freq=25

