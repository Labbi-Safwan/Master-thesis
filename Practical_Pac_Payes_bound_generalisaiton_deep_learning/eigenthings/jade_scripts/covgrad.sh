#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=SGDSpectrum

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
python3 ../spectrum.py --spectrum_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD_noschedule/lr=0.1_wd=0.0005/covgrad-spectrum-00100.npz --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --ckpt /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/SGD_noschedule/lr=0.1_wd=0.0005/checkpoint-00100.pt --iters 10
