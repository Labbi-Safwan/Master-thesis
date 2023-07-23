#!/bin/bash

# THIS IS RUN ON RAPID 

# run the application

# AdamWSWA
# Weight decay = 1e-3
#python3 run_sgd.py --dir /nfs/home/dgranziol/curvature/out/VGG16/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --lr_init 0.03 --epochs 300 --save_freq=25 --eval_freq=1

python3 run_sgd.py --dir /nfs/home/dgranziol/curvature/ckpts/c10/VGG16/ --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16basic --lr_init 0.05 --epochs 300 --save_freq=25 --eval_freq=1

python3 run_sgd.py --dir /nfs/home/dgranziol/kfac-curvature/out/ --dataset CIFAR10 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16basic --lr_init 0.05 --epochs 300 --save_freq=25 --eval_freq=1

#shallow
python3 run_sgd.py --dir /homes/48/diego/kfac-curvature/ckpts/ --dataset CIFAR100 --data_path /homes/48/diego/kfac-curvature/data/ --model VGG16 --lr_init 0.05 --epochs 300 --save_freq=25 --eval_freq=1
