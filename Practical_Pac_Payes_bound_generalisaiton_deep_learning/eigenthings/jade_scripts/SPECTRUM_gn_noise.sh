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


# Curveball
# Weight decay = 0

python3 spectrum.py --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16 --batch_size=128 --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/checkpoint-00000.pt --iters 100 --num_samples 128 --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/hess_noise-00000 --curvature_matrix gn_noise

python3 spectrum.py --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16 --batch_size=128 --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/checkpoint-00100.pt --iters 100 --num_samples 128 --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/hess_noise-00100 --curvature_matrix gn_noise

python3 spectrum.py --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16 --batch_size=128 --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/checkpoint-00200.pt --iters 100 --num_samples 128 --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/hess_noise-00200 --curvature_matrix gn_noise

python3 spectrum.py --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16 --batch_size=128 --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/checkpoint-00300.pt --iters 100 --num_samples 128 --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16/SGD_noschedule/spectrum/run1/hess_noise-00300 --curvature_matrix gn_noise
