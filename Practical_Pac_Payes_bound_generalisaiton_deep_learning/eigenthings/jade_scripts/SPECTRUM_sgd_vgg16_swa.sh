#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=SGDSWASpectrum

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


python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00000.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00000.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00050.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00050.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00100.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00100.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00150.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00150.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00200.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00200.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00250.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00250.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/SGDSWA_noschedule/spectrum-00299.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/checkpoint-00300.pt --iters 50 --swag
