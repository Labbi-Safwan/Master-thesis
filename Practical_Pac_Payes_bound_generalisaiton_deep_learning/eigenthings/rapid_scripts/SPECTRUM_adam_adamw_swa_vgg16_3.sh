#!/bin/bash

# THIS IS RUN ON RAPID

source activate curvature

# AdamWSWA

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/spectrum-00300.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/swag-00300.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/spectrum-00200.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/swag-00200.pt --iters 50 --swag

