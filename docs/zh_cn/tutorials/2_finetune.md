# 教程 2：如何微调模型

本教程介绍如何使用预训练模型在其他数据集上进行微调。

<!-- TOC -->

- [教程 2：如何微调模型](#教程-2如何微调模型)
  - [概要](#概要)
  - [修改 Head](#修改-head)
  - [修改数据集](#修改数据集)
  - [修改训练策略](#修改训练策略)
  - [使用预训练模型](#使用预训练模型)

<!-- TOC -->

## 概要

对新数据集上的模型进行微调需要进行两个步骤：

1. 增加对新数据集的支持。详情请见 [教程 3：如何增加新数据集](3_new_dataset.md)
2. 修改配置文件。这部分将在本教程中做具体讨论。

例如，如果用户想要微调 Kinetics-400 数据集的预训练模型到另一个数据集上，如 UCF101，则需要注意 [配置文件](1_config.md) 中 Head、数据集、训练策略、预训练模型四个部分，下面分别介绍。

## 修改 Head

`cls_head` 中的 `num_classes` 参数需改为新数据集中的类别数。
预训练模型中，除了最后一层外的权重都会被重新利用，因此这个改动是安全的。
例如，UCF101 拥有 101 类行为，因此需要把 400 (Kinetics-400 的类别数) 改为 101。

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

其中， `pretrained='torchvision://resnet50'` 表示通过 ImageNet 预训练权重初始化 backbone。
然而，模型微调时的预训练权重一般通过 `load_from`（而不是 `pretrained`）指定。

## 修改数据集

MMAction2 支持 UCF101, Kinetics-400, Moments in Time, Multi-Moments in Time, THUMOS14,
Something-Something V1&V2, ActivityNet 等数据集。
用户可将自建数据集转换已有数据集格式。
对动作识别任务来讲，MMAction2 提供了 `RawframeDataset` 和 `VideoDataset` 等通用的数据集读取类，数据集格式相对简单。
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

## 修改训练策略

通常情况下，设置较小的学习率，微调模型少量训练批次，即可取得较好效果。

```python
# 优化器
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)  # 从 0.01 改为 0.005
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# 学习策略
lr_config = dict(policy='step', step=[20, 40]) # step 与 total_epoch 相适应
total_epochs = 50 # 从 100 改为 50
checkpoint_config = dict(interval=5)
```

## 使用预训练模型

若要将预训练模型用于整个网络（主干网络设置中的 `pretrained`，仅会在主干网络模型上加载预训练参数），可通过 `load_from` 指定模型文件路径或模型链接，实现预训练权重导入。
MMAction2 在 `configs/_base_/default_runtime.py` 文件中将 `load_from=None` 设为默认。由于配置文件的可继承性，用户可直接在下游配置文件中设置 `load_from` 的值来进行更改。

```python
# 将预训练模型用于整个 TSN 网络
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'  # 模型路径可以在 model zoo 中找到
```
