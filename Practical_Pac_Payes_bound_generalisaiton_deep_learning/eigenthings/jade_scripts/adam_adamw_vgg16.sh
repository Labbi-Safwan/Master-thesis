#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=PreResNet110

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

# Adam-L2
# Weight decay = 0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Adam/Adam_wd0/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0 --epochs 300 --save_freq=5

# Weight decay = 1e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Adam/Adam_wd1e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-5 --epochs 300 --save_freq=5

# Weight decay = 5e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Adam/Adam_wd5e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-5 --epochs 300 --save_freq=5

# Weight decay = 1e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Adam/Adam_wd1e-4/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-4 --epochs 300 --save_freq=5

# Weight decay = 5e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Adam/Adam_wd5e-4/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-4 --epochs 300 --save_freq=5

# Weight decay = 1e-3
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/Adam/Adam_wd1e-3/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-3 --epochs 300 --save_freq=5

# AdamW
# Weight decay = 0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamW/AdamW_wd0/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 0 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamW/AdamW_wd1e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-5 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamW/AdamW_wd5e-5/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-5 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamW/AdamW_wd1e-4/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-4 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 5e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamW/AdamW_wd5e-4/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 5e-4 --epochs 300 --save_freq=5 --decoupled_wd

# Weight decay = 1e-3
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamW/AdamW_wd1e-3/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --lr_init 0.0005 --wd 1e-3 --epochs 300 --save_freq=5 --decoupled_wd

