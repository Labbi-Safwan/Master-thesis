#!/bin/bash

# THIS IS RUN ON RAPID

source activate curvature

# AdamW
python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/spectrum-00100.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-1/checkpoint-00100.pt --iters 50

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/spectrum-00200.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-1/checkpoint-00200.pt --iters 50

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/spectrum-00300.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamW/AdamW_wd5e-1/checkpoint-00300.pt --iters 50

# AdamWSWA
python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/spectrum-00100.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/checkpoint-00100.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/spectrum-00200.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/checkpoint-00200.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/spectrum-00300.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/AdamWSWA/AdamW_wd5e-1/checkpoint-00300.pt --iters 50 --swag
