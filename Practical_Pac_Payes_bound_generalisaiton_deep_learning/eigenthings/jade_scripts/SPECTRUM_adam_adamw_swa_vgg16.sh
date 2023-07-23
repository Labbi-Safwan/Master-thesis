#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=AdamSWASpectrum

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


python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00000.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00000.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00000.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00050.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00050.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00050.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00100.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00100.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00100.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00150.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00150.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00150.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00200.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00200.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00200.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00250.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00250.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00250.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/spectrum-00299.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamSWA_noschedule/Adam_wd5e-4/checkpoint-00300.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamSWA_noschedule/basis-00300.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00000.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00000.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00000.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00050.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00050.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00050.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00100.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00100.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00100.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00150.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00150.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00150.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00200.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00200.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00200.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00250.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00250.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00250.npz --iters 50 --swag

python3 spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/spectrum-00299.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/data/ --model VGG16BN --ckpt /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/VGG16BN/AdamWSWA/Adam_wd5e-4/checkpoint-00300.pt --basis_path  /jmain01/home/JAD017/sjr01/xxw68-sjr01/curvature/out/spectrum/VGG16BN/AdamWSWA/basis-00300.npz --iters 50 --swag
