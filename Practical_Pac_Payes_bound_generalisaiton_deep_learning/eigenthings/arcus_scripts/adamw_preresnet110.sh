#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=100:59:59
# set name of job
#SBATCH --job-name=Adam

#choose partition
#SBATCH --partition=small

# run the application
# module load python3/anaconda
source activate curvature

# Weight decay = 5e-3
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd5e-3/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 5e-3 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25

# Weight decay = 1e-2
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd1e-2/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 1e-2 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25

# Weight decay = 5e-2
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd5e-2/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 5e-2 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25

# Weight decay = 1e-1
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd1e-1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 1e-1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25

# Weight decay = 5e-1
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd5e-1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 5e-1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25

# Weight decay = 1
python3 run_adam.py --dir /data/engs-bayesian-machine-learning/sann5476/curvature/out/PreResNet110/AdamW/AdamW_wd1/ --dataset CIFAR100 --data_path /data/engs-bayesian-machine-learning/sann5476/curvature/data/ --model PreResNet110 --lr_init 0.001 --wd 1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25
