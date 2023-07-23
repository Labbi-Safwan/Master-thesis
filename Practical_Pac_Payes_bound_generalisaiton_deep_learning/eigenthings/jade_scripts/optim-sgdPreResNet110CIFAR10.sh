#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=sgd
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.001 --wd=1e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.001 --wd=5e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.001 --wd=0.0001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.001 --wd=0.0005
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.001 --wd=0.001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.005 --wd=1e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.005 --wd=5e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.005 --wd=0.0001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.005 --wd=0.0005
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.005 --wd=0.001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=1e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=5e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=0.0001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=0.0005
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=0.001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=1e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=5e-05
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=0.0001
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=0.0005
python3 run_sgd.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=0.001
