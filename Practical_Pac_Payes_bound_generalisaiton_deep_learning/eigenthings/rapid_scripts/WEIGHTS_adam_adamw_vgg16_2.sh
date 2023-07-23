#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=AdamSpectrum

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

# AdamW
python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-3/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-3/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd1e-2/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd1e-2/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-2/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-2/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd1e-1/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd1e-1/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-1/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-1/weight_norms.npz

# AdamWSWA
python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-3/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-3/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1e-2/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1e-2/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-2/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-2/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1e-1/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd1e-1/weight_norms.npz

python3 compute_weight_norm.py --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --epochs 300 --dir /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/ --save_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/weight_norms.npz
