#!/bin/bash

# THIS IS RUN ON RAPID

source activate curvature

# KFACSWA

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA/spectrum-00275.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA/KFAC_wd1e-4/swag-00275.pt --iters 50 --swag

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA/spectrum-00275-trainoff.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/KFACSWA/KFAC_wd1e-4/swag-00275.pt --iters 50 --swag --bn_train_mode_off

# KFAC

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/spectrum-00275.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd1e-4/checkpoint-00275.pt --iters 50 

python3 spectrum.py --spectrum_path /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/spectrum-00275-trainoff.npz --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model VGG16BN --ckpt /nfs/home/xingchenw/curvature/out/VGG16BN/KFAC/KFAC_wd1e-4/checkpoint-00275.pt --iters 50  --bn_train_mode_off

