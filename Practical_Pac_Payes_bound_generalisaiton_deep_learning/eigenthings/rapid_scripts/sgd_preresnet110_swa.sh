#!/bin/bash

source activate curvature

# SGD_noschedule-L2
# Weight decay = 0
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out/PreResNet110/SGDSWA_noschedule/SGD_wd0/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --lr_init 0.1 --wd 0 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.05

# Weight decay = 1e-5
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out/PreResNet110/SGDSWA_noschedule/SGD_wd1e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --lr_init 0.1 --wd 1e-5 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.05

# Weight decay = 5e-5
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out/PreResNet110/SGDSWA_noschedule/SGD_wd5e-5/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --lr_init 0.1 --wd 5e-5 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.05

# Weight decay = 1e-4
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out/PreResNet110/SGDSWA_noschedule/SGD_wd1e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --lr_init 0.1 --wd 1e-4 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.05

# Weight decay = 5e-4 - this is not needed as we already have data for wd = 5e-4
# python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out/PreResNet110/SGD_noschedule/SGD_wd5e-4/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --lr_init 0.0005 --wd 5e-4 --epochs 300 --save_freq=5

# Weight decay = 1e-3
python3 run_sgd.py --dir /nfs/home/xingchenw/curvature/out/PreResNet110/SGDSWA_noschedule/SGD_wd1e-3/ --dataset CIFAR100 --data_path /nfs/home/xingchenw/curvature/data/ --model PreResNet110 --lr_init 0.1 --wd 1e-3 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.05
