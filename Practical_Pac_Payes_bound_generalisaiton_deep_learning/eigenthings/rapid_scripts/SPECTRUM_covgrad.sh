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
source activate diegorubin


# SGD_noschedule
python3 spectrum.py --spectrum_path  /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/cov_grad.npz --dataset CIFAR100 --data_path  /nfs/home/xingchenw/curvature/data/ --model VGG16 --ckpt  /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt --iters 50 --curvature_matrix='covgrad'

python3 spectrum.py --curvature_matrix='' --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt --iters=5