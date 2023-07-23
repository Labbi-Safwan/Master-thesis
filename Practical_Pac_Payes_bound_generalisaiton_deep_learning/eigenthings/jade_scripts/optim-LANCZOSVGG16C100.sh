#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=lanczos
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 1 --lanczos_beta=10 --wd=0
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.1 --lanczos_beta=1 --wd=0
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 1 --lanczos_beta=50 --wd=0
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.1 --lanczos_beta=5 --wd=0
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.01 --lanczos_beta=0.5 --wd=0
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.001 --lanczos_beta=0.05 --wd=0
