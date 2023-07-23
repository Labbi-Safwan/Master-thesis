#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=4PreResNet164

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

# Weight decay = 0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamSWA_noschedule/Adam_wd0_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 0 --epochs 300 --save_freq=5  --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 1e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamSWA_noschedule/Adam_wd1e-5_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 1e-5 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 5e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamSWA_noschedule/Adam_wd5e-5_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 5e-5 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 1e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamSWA_noschedule/Adam_wd1e-4_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 1e-4 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 5e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamSWA_noschedule/Adam_wd5e-4_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 5e-4 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 1e-3
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamSWA_noschedule/Adam_wd1e-3_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 1e-3 --epochs 300 --save_freq=5 --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# AdamW
# Weight decay = 0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamWSWA/AdamW_wd0_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 0 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 1e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamWSWA/AdamW_wd1e-5_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 1e-5 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 5e-5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamWSWA/AdamW_wd5e-5_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 5e-5 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 1e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamWSWA/AdamW_wd1e-4_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 1e-4 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 5e-4
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamWSWA/AdamW_wd5e-4_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 5e-4 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

# Weight decay = 1e-3
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/PreResNet164/AdamWSWA/AdamW_wd1e-3_lr_25e-4_slr_125e-5/ --dataset CIFAR10 --data_path /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model PreResNet164 --lr_init 0.01 --wd 1e-3 --epochs 300 --save_freq=5 --decoupled_wd --swag --swag_rank=20 --swag_start=161 --swag_lr=0.005

