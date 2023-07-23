#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=adam
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0001
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.0003
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.03
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.03
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.03
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.03
python3 run_adam.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.03
