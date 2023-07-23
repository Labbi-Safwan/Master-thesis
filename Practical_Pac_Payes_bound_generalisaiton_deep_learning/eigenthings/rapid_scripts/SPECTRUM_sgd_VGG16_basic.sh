#!/bin/bash

# THIS IS RUN ON RAPID


# SGDSWA_noschedule

#python3 spectrum.py --curvature_matrix=nonggn --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/GN-00225.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/GN-00225  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model PreResNet110 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/checkpoint-00225.pt



python3 ../spectrum.py --curvature_matrix=hessian --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/Hess-eval-00225.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/Hess-eval-00225  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model PreResNet110 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/checkpoint-00225.pt --seed=1 --iters=100
python3 ../spectrum.py --curvature_matrix=hessian --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c10/PreResNet110/SGDOPT/Hess-eval-00225.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c10/PreResNet110/SGDOPT/Hess-eval-00225  --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model PreResNet110 --ckpt /nfs/home/dgranziol/curvature/ckpts/c10/PreResNet110/SGDOPT/checkpoint-00225.pt --seed=1 --iters=100

python3 ../spectrum.py --curvature_matrix=hessian --bn_train_mode --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/Hess-train-00225.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/Hess-train-00225  --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model PreResNet110 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/PreResNet110/SGDOPT/checkpoint-00225.pt --seed=1 --iters=100
python3 ../spectrum.py --curvature_matrix=hessian --bn_train_mode --spectrum_path /nfs/home/dgranziol/curvature/ckpts/c10/PreResNet110/SGDOPT/Hess-train-00225.npz --basis_path /nfs/home/dgranziol/curvature/ckpts/c10/PreResNet110/SGDOPT/Hess-train-00225  --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model PreResNet110 --ckpt /nfs/home/dgranziol/curvature/ckpts/c10/PreResNet110/SGDOPT/checkpoint-00225.pt --seed=1 --iters=100

python3 ../spectrum.py --curvature_matrix=nonggn --spectrum_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG16basicSGD/lr=0.05_wd=0.0005/nonGN-00300.npz --basis_path /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG16basicSGD/lr=0.05_wd=0.0005/nonGN-00300  --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16basic --ckpt /nfs/home/dgranziol/kfac-curvature/out/CIFAR10/VGG16basicSGD/lr=0.05_wd=0.0005/checkpoint-00300.pt --iters=100
