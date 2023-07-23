#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

#SBATCH --gres=gpu:1
# set name of job
#SBATCH --job-name=AdamWNeXt

#choose partition
#SBATCH --partition=small

# run the application
module load python3/anaconda
source activate curvature

# Weight decay = 1e-2
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd1e-2/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1e-2 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25 --use_test

# Weight decay = 5e-2
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd5e-2/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 5e-2 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25 --use_test

# Weight decay = 1e-1
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd1e-1/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1e-1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25 --use_test

# Weight decay = 5e-1
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd5e-1/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 5e-1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25  --use_test

# Weight decay = 0.25
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd0.25/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 0.25 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25  --use_test

# Weight decay = 5e-1
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/ResNeXt29CIFAR/AdamW/AdamW_wd1/ --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model ResNeXt29CIFAR --lr_init 0.001 --wd 1 --epochs 300 --eval_freq=5 --decoupled_wd --save_freq=25  --use_test

