﻿#!/bin/sh
python test.py --data-root /home/ziqi/data/AI_Yaogan/image_B --batch-size 32 --backbone resnet101 --model deeplabv3plus --load-from /home/ziqi/data/AI_Yaogan/result/resnet101_deeplabv3plus/models/resnet101_deeplabv3plus_best.pth --tta
python test.py --data-root /home/ziqi/data/AI_Yaogan/image_B --batch-size 32 --backbone resnext50 --model deeplabv3plus --load-from /home/ziqi/data/AI_Yaogan/result/resnext50_deeplabv3plus/models/resnext50_deeplabv3plus_best.pth --tta