#!/bin/bash

# THIS IS RUN ON RAPID

# SGDSWA_noschedule

python3 diagspec.py --dataset CIFAR100 --data_path /nfs/home/dgranziol/curvature/data/ --model VGG16 --ckpt /nfs/home/dgranziol/curvature/ckpts/c100/VGG16/SGDOPT/checkpoint-00300.pt --use_test
