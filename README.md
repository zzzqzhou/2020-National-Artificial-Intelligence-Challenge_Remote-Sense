# NAIC遥感竞赛

## 实验设置
初赛使用了`ResNet101`和`ResNeXt50`这两个模型作为backbone，采用了PyTorch官方提供的在ImageNet上的预训练模型，预训练模型下载链接见下方。分割模型的head采用了`DeepLabV3+`。在1块2080Ti上进行训练，batch设置大小为32，训练轮数共120个epoch，采用SGD作为优化器，动量参数momentum为0.9，权重衰减weight_decay为0.0001，初始学习率0.002，采用lr_poly策略调整学习率，其中power参数设置为0.9。更多详细的参数设定见`train.py`和`train.sh`。

测试阶段使用Test Time Augmentation，将1张原图进行翻转、旋转后共6张图片分别输入模型得到6个预测结果，将6个预测结果求平均后得到最终的预测概率图。


## 预训练模型下载：
预训练模型下载后存放在`./pretrained_models`文件夹下
<a href="https://pan.baidu.com/s/1NMQplJgiv7tE_9dRNXmpzA">模型</a> 提取码：94sc

## 结果模型下载：
训练好的模型下载后直接存放在当前文件夹下
<a href="https://pan.baidu.com/s/1XCdp66a_upzk_rXP6Th2Ng">模型</a> 提取码：ueiw 

## 训练命令:
训练产生两个模型：`ResNet101_DeepLabV3+`和`ResNeXT50_DeepLabV3+`，模型保存在`./result`文件夹下
```
bash train.sh
```
## 测试命令
生成`ResNet101_DeepLabV3+`和`ResNeXT50_DeepLabV3+`的预测结果，生成的预测图片分别保存在`./test_B_rn101_dlv3p`和`./test_B_rnx50_dlv3p`下，将相应文件夹下的`results/`文件夹打包即可提交。
```
bash test.sh
```
## Ensemble命令
生成`ResNet101_DeepLabV3+`和`ResNeXT50_DeepLabV3+`融合的预测结果，生成的预测图片保存在`./test_B_ensemble`下，将该文件夹下的`results/`文件夹打包即可提交。
```
bash ensemble.sh
```