#!/bin/bash

# THIS IS RUN ON RAPID


# SGDSWA_noschedule

#python3 spectrum.py --curvature_matrix=gn --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt


python3 ../spectrum.py --curvature_matrix=gn --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300-2.npz  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt --seed=2 --iters=100
python3 ../spectrum.py --curvature_matrix=gn --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300-3.npz  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt --seed=3 --iters=100
