#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=Curveball

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=xingchen.wan@outlook.com

# run the application
module load python3/anaconda
source activate curvature


python3 experiments/lanczos/spectrum.py --dataset=CIFAR10 --data_path=/nfs/home/dgranziol/curvature/data/ --model=VGG16 --subsample_seed=1 --seed=1 --ckpt=./ckpts/c10/VGG16/swag/run1/checkpoint-00300.pt --iters=100 --basis_path=./ckpts/c10/VGG16/swag/run1/300-full --spectrum_path=./ckpts/c10/VGG16/swag/run1/300-full.npz


# Curveball
# Weight decay = 0
python3 run_curveball.py --dir /nfs/home/dgranziol/curvature/curveball/ --dataset CIFAR100 --data_path=/nfs/home/dgranziol/curvature/data/  --model VGG16BN --lr_init -1 --wd 0 --epochs 300 --save_freq=5 --beta_init -1 --lambda_init 10

python3 run_KFAC.py --dir=./ckpts/c10/vgg16/KFAC/run1/ --dataset=CIFAR10 --data_path=/home/tgaripov/projects/data/  --use_test --model=VGG16 --epochs=300 --lr_init -1 --wd 0 --epochs 300 --save_freq=5



# Weight decay = 1e-5
python3 run_curveball.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Curveball/Curveball_wd1e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init -1 --wd 1e-5 --epochs 300 --save_freq=5--beta_init -1 --lambda_init 10

# Weight decay = 5e-5
python3 run_curveball.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Curveball/Curveball_wd5e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init -1 --wd 5e-5 --epochs 300 --save_freq=5 --beta_init -1 --lambda_init 10

# Weight decay = 1e-4
python3 run_curveball.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Curveball/Curveball_wd1e-4/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init -1 --wd 1e-4 --epochs 300 --save_freq=5 --beta_init -1 --lambda_init 10

# Weight decay = 5e-4 - this is not needed as we already have data for wd = 5e-4
python3 run_curveball.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Curveball/Curveball_wd5e-4/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init -1 --wd 5e-4 --epochs 300 --save_freq=5 --beta_init -1 --lambda_init 10

# Weight decay = 1e-3
python3 run_curveball.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Curveball/Curveball_wd1e-3/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init -1 --wd 1e-3 --epochs 300 --save_freq=5 --beta_init -1 --lambda_init 10
