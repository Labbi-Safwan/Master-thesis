#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

# AdamWSWA
# Weight decay = 1e-3
python3 run_sgd.py --dir /nfs/home/dgranziol/curvature/out/VGG11/C100/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG11 --lr_init 0.03 --epochs 300 --save_freq=25 --eval_freq=1


python3 autodamping_KFAC.py --dir /nfs/home/dgranziol/curvature/out/VGG16/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 0.003 --epochs 300 --save_freq=25 --eval_freq=1
python3 shrinkage_adam.py --dir /nfs/home/dgranziol/curvature/out/VGG16/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 0.001 --epochs 300 --save_freq=25 --eval_freq=1
python3 layer_shrinkage_sgd.py --dir /nfs/home/dgranziol/curvature/out/VGG16/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16BN --lr_init 0.1 --epochs 300 --save_freq=25 --eval_freq=1

python3 lr_sgd.py --dir /nfs/home/dgranziol/curvature/out/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16BN --lr_init 0.1 --epochs 300 --save_freq=25 --eval_freq=1 --lr_freq=1


python3 run_sgd.py --dir /nfs/home/dgranziol/kfac-curvature/out/ --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG6 --lr_init 0.1 --epochs 300 --save_freq=25 --eval_freq=1