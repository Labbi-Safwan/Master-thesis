#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=gnlanc
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin

python3 ../run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 1 --epochs 100 --save_freq=25 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn' --lanczos_beta=10 --no_schedule
#python3 ../run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.1 --epochs 100 --save_freq=25 --eval_freq=1 --lanczos_steps=30 --matrix_type='gn' --lanczos_beta=1
