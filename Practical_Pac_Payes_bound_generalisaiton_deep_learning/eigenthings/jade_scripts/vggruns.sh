#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=vggruns
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=0
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.035 --wd=0
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.02 --wd=0

python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=5e-4
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.035 --wd=5e-4
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.02 --wd=5e-4

python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.1 --wd=0
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=0
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.035 --wd=0
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.02 --wd=0

python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.1 --wd=5e-4
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.05 --wd=5e-4
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.035 --wd=5e-4
python3 ../run_sgd.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.02 --wd=5e-4

