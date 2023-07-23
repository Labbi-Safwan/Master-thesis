#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vgg16landscape

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

python3 ../spectrum.py --spectrum_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/300full.npz --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model PreResNet110 --ckpt /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/checkpoint-00300.pt --basis_path  /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/300full --iters 100

python3 ../hess_landscape_1d.py --dataset=CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --basis_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/300full --seed=1 --save_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/hess_loss_results-d=1.npz --dist=1 --use_test
python3 ../hess_landscape_1d.py --dataset=CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --basis_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/300full --seed=1 --save_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/hess_loss_results-d=2.npz --dist=2 --use_test
python3 ../hess_landscape_1d.py --dataset=CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --basis_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/300full --seed=1 --save_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/hess_loss_results-d=0p5.npz --dist=0.5 --use_test

python3 ../hess_landscape_1d.py --dataset=CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --basis_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/GN-100-00300 --seed=1 --save_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/GN_loss_results-d=1.npz --dist=1 --use_test
python3 ../hess_landscape_1d.py --dataset=CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --basis_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/GN-100-00300 --seed=1 --save_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/GN_loss_results-d=2.npz --dist=2 --use_test
python3 ../hess_landscape_1d.py --dataset=CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --basis_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/GN-100-00300 --seed=1 --save_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/VGG16/SGDOPT/GN_loss_results-d=0p5.npz --dist=0.5 --use_test
