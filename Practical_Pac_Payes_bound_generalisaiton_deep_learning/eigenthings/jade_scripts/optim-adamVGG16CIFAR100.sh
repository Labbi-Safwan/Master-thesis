#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=adam
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 3e-05
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
