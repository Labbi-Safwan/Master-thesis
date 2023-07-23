#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=bnautodamp
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin

python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.3 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=0
python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.01 --damping 0.03 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=0

python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.3 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=0 --ma
 python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.01 --damping 0.03 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=0 --ma

python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.3 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=5
python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.01 --damping 0.03 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=5 --ma

python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.3 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=50
python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.1 --damping 0.3 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=50 --ma
python3 ../autodamping_KFAC.py --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16BN --lr_init 0.01 --damping 0.03 --epochs 300 --save_freq=25 --eval_freq=1 --no_schedule --wd_start=50 --ma


