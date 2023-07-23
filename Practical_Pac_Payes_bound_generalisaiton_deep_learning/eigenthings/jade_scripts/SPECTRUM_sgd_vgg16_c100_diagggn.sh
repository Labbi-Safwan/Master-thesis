#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=bkpkSpc

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
source activate diegorubin

python3 spectrum.py --curvature_matrix=gn_diag --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG6/SGD_noschedule/lr=0.05_wd=0.0005/diag_gn-00300.npz  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG6/SGD_noschedule/lr=0.05_wd=0.0005/checkpoint-00300.pt

python3 spectrum.py --curvature_matrix=hessian_diag --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG6/SGD_noschedule/lr=0.05_wd=0.0005/diag_hessian-00300.npz  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG6/SGD_noschedule/lr=0.05_wd=0.0005/checkpoint-00300.pt

python3 spectrum.py --curvature_matrix=gn_diag_mc --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG6/SGD_noschedule/lr=0.05_wd=0.0005/diag_gn_mc-00300.npz  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG6/SGD_noschedule/lr=0.05_wd=0.0005/checkpoint-00300.pt

