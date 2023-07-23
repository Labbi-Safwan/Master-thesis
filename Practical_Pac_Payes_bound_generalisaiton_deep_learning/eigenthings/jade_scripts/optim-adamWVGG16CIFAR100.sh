#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=adamW
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05 --decoupled_wd --wd=0.333333333333
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05 --decoupled_wd --wd=1.66666666667
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05 --decoupled_wd --wd=3.33333333333
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05 --decoupled_wd --wd=16.6666666667
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05 --decoupled_wd --wd=33.3333333333
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001 --decoupled_wd --wd=0.1
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001 --decoupled_wd --wd=0.5
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001 --decoupled_wd --wd=1.0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001 --decoupled_wd --wd=5.0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001 --decoupled_wd --wd=10.0
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003 --decoupled_wd --wd=0.0333333333333
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003 --decoupled_wd --wd=0.166666666667
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003 --decoupled_wd --wd=0.333333333333
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003 --decoupled_wd --wd=1.66666666667
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003 --decoupled_wd --wd=3.33333333333
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --decoupled_wd --wd=0.001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --decoupled_wd --wd=0.005
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --decoupled_wd --wd=0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --decoupled_wd --wd=0.05
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --decoupled_wd --wd=0.1
