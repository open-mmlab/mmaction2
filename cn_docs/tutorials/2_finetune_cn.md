# 教程 2：如何微调模型

本教程为用户提供使用预训练模型在其他数据集上进行微调的说明，以便获得更好的性能。

<!-- TOC -->

- [概要](#概要)
- [修改 Head](#修改-Head)
- [修改数据集](#修改数据集)
- [修改训练调度](#修改训练调度)
- [使用预训练模型](#使用预训练模型)

<!-- TOC -->

## 概要

对新数据集上的模型进行微调需要进行两个步骤：

1. 增加对新数据集的支持。详情请见 [教程 3：如何增加新数据集](3_new_dataset_cn.md)
2. 修改配置文件。这部分将在本教程中做具体讨论。

例如，如果用户想要微调 Kinetics-400 数据集的预训练模型到另一个数据集上，如 UCF101，则需要注意[配置文件](1_config_cn.md)中的这四个部分。

## 修改 Head

`cls_head` 中的 `num_classes` 参数需要被修改为新数据集中的类别数。
预训练模型中，除了最后一层外的权重都会被重新利用，因此这个改动是安全的。
对于 UCF101，它有 101 个类，因此需要把 400 (Kinetics-400 的类别数) 改为 101。

```python
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=101,   # 从 400 改为 101
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips=None))
```

注意，这里的 `pretrained='torchvision://resnet50'` 用于初始化 backbone，用于继承 ImageNet 预训练的权重来训新模型。
然而，这个设置和微调模型没有关系。预训练的权重的加载通过 `load_from` 来指定。

## 修改数据集

MMAction2 支持 UCF101, Kinetics-400, Moments in Time, Multi-Moments in Time, THUMOS14,
Something-Something V1&V2, ActivityNet 等数据集。
用户可将他们特定的数据集调整为上面支持的数据集格式之一。
对动作识别任务来讲，数据集格式相对简单。MMAction2 提供了 `RawframeDataset` 和 `VideoDataset` 等通用的数据集读取类。
以 `UCF101` 和 `RawframeDataset` 为例，

```python
# 数据集设置
dataset_type = 'RawframeDataset'
data_root = 'data/ucf101/rawframes_train/'
data_root_val = 'data/ucf101/rawframes_val/'
ann_file_train = 'data/ucf101/ucf101_train_list.txt'
ann_file_val = 'data/ucf101/ucf101_val_list.txt'
ann_file_test = 'data/ucf101/ucf101_val_list.txt'
```

## 修改训练调度

微调模型通常需要较小的学习率和较少的训练时间。

```python
# 优化器
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)  # 从 0.01 改为 0.005
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# 学习策略
lr_config = dict(policy='step', step=[40, 80]) # step 与 total_epoch 相适应
total_epochs = 50 # 从 100 改为 50
checkpoint_config = dict(interval=5)
```

## 使用预训练模型

若要将预训练模型用于整个网络，需要在新的配置中，于 `load_from` 处指明预训练模型的链接。

```python
# 将预训练模型用于整个 TSN 网络
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'  # model path can be found in model zoo
```
