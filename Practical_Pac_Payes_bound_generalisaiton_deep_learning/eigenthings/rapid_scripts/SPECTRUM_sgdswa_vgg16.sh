#!/bin/bash

# THIS IS RUN ON RAPID

source activate curvature

# SGDSWA_noschedule

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/SGDSWA_noschedule/spectrum-00300.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/swag-00300.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/SGDSWA_noschedule/spectrum-00200.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/SGDSWA_noschedule/SGD_wd5e-4/swag-00200.pt --iters 50 --swag
