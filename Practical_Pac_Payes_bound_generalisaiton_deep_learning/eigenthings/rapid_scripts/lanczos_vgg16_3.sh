#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

# AdamWSWA
# Weight decay = 1e-3
python3 ../run_lanczos.py --dir /nfs/home/dgranziol/curvature/out --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 1 --epochs 100 --save_freq=25 --eval_freq=1 --lanczos_steps=30 --matrix_type='hessian' --lanczos_beta=50
python3 ../run_lanczos.py --dir /nfs/home/dgranziol/curvature/out --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 1 --epochs 100 --save_freq=25 --eval_freq=1 --lanczos_steps=30 --matrix_type='gn' --lanczos_beta=50

python3 ../run_lanczos.py --dir /nfs/home/dgranziol/curvature/out --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 1 --epochs 100 --save_freq=25 --eval_freq=1 --lanczos_steps=30 --matrix_type='hessian' --lanczos_beta=50
python3 ../run_lanczos.py --dir /nfs/home/dgranziol/curvature/out --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 1 --epochs 100 --save_freq=25 --eval_freq=1 --lanczos_steps=30 --matrix_type='gn' --lanczos_beta=50

