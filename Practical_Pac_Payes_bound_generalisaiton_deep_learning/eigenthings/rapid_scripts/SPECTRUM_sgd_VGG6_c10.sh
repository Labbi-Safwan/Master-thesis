#!/bin/bash

# THIS IS RUN ON RAPID


# SGDSWA_noschedule

#python3 spectrum.py --curvature_matrix=gn --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/GN-00300  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt

python3 ../spectrum.py --curvature_matrix=gn_diag --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/diag_gn-00300.npz  --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/checkpoint-00300.pt

python3 ../spectrum.py --curvature_matrix=gn_diag_mc --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/diag_gn_mc-00300.npz --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/checkpoint-00300.pt

python3 ../spectrum.py --curvature_matrix=gn --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/GN-00300.npz --basis_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/GN-00300  --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/checkpoint-00300.pt --iters=100

python3 ../spectrum.py --curvature_matrix=hessian --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/Hess-00300.npz --basis_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/Hess-00300  --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG6 --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG6/SGD_noschedule/lr=0.01_wd=0.0005/checkpoint-00300.pt --iters=100

