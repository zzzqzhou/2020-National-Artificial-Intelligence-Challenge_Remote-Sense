#!/bin/sh
python train.py --epochs 120 --lr 0.001 --backbone resnet101 --model deeplabv3plus --batch-size 16 --data-root ./train_data --log-dir ./result/resnet101_deeplabv3plus --pretrained
python train.py --epochs 120 --lr 0.001 --backbone resnext50 --model deeplabv3plus --batch-size 16 --data-root ./train_data --log-dir ./result/resnext50_deeplabv3plus --pretrained