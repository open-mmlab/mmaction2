# 模型微调

本教程提供了使用预训练模型在其他数据集上进行微调的指导。通过微调，可以获得更好的性能。

- [模型微调](#模型微调)
  - [概述](#概述)
  - [选择模板配置](#选择模板配置)
  - [修改 Head](#修改-head)
  - [修改数据集](#修改数据集)
  - [修改训练计划](#修改训练计划)
  - [使用预训练模型](#使用预训练模型)
  - [开始训练](#开始训练)

## 概述

在新数据集上进行模型微调有两个步骤。

1. 添加对新数据集的支持。请参考[准备数据集](prepare_dataset.md)和[自定义数据集](../advanced_guides/customize_dataset.md)。
2. 修改配置文件。本教程将讨论这一部分。

## 选择模板配置

这里我们以 `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` 为例。我们首先将该配置文件复制到同一文件夹，并将其重命名为 `tsn_ucf101.py`，然后需要注意配置中的四个部分，具体来说，为不存在的键添加新键，并修改现有键的原始键。

## 修改 Head

`cls_head` 中的 `num_classes` 需要更改为新数据集的类别数。预训练模型的权重会被重用，除了最后的预测层。因此，更改类别数是安全的。在我们的例子中，UCF101 有 101 个类别。所以我们将其从 400（Kinetics-400 的类别数）改为 101。

```python
# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=101  # 将 400 修改为 101
        ))
```

## 修改数据集

MMAction2 支持 UCF101、Kinetics-400、Moments in Time、Multi-Moments in Time、THUMOS14、Something-Something V1&V2、ActivityNet 数据集。用户可能需要将上述其中一个数据集适应到他们的特殊数据集上。你可以参考[准备数据集](prepare_dataset.md)和[自定义数据集](../advanced_guides/customize_dataset.md)了解更多细节。在我们的例子中，UCF101 已经由各种数据集类型支持，例如 `VideoDataset`，因此我们将配置修改如下。

```python
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos_train/'
data_root_val = 'data/ucf101/videos_val/'
ann_file_train = 'data/ucf101/ucf101_train_list.txt'
ann_file_val = 'data/ucf101/ucf101_val_list.txt'
```

## 修改训练计划

微调通常需要较小的学习率和较少的训练周期。

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # 将 100 修改为 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # 将 100 修改为 50
        by_epoch=True,
        milestones=[20, 40],  # 修改 milestones
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # 将 0.01 修改为 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))
```

## 使用预训练模型

为了在整个网络上使用预训练模型，新配置文件在 `load_from` 中添加了预训练模型的链接。我们在 `configs/_base_/default_runtime.py` 中设置 `load_from=None` 作为默认值，并且根据[继承设计](config.md)，用户可以通过在其配置中设置 `load_from` 来直接更改它。

```python
# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'  # 模型路径可以在模型库中找到
```

## 开始训练

现在，我们已经完成了微调的配置文件，如下所示：

```python
_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=101  # 将 400 修改为 101
        ))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos_train/'
data_root_val = 'data/ucf101/videos_val/'
ann_file_train = 'data/ucf101/ucf101_train_list.txt'
ann_file_val = 'data/ucf101/ucf101_val_list.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # 将 100 修改为 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # 将 100 修改为 50
        by_epoch=True,
        milestones=[20, 40],  # 修改 milestones
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # 将 0.01 修改为 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)

# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'

```

另一种更简单的方法是继承 kinetics400 配置，并只指定修改的键。请确保自定义配置与 `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` 在同一个文件夹中。

```python
_base_ = [
    'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'  # 继承模板配置
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=101))  # 将 400 修改为 101


# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos_train/'
data_root_val = 'data/ucf101/videos_val/'
ann_file_train = 'data/ucf101/ucf101_train_list.txt'
ann_file_val = 'data/ucf101/ucf101_val_list.txt'

train_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root)))
val_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)))
test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # 将 100 修改为 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # 将 100 修改为 50
        by_epoch=True,
        milestones=[20, 40],  # 修改 milestones
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # 将 0.01 修改为 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'

```

你可以使用以下命令在你的数据集上微调模型。

```shell
python tools/train.py ${CONFIG_FILE} [可选参数]
```

例如：在确定性选项下，在 Kinetics-400 数据集上训练 TSN 模型。

```shell
python tools/train.py configs/recognition/tsn/tsn_ucf101.py  \
  --seed=0 --deterministic
```

更多细节，请参考[训练和测试教程](train_test.md)中的**训练**部分。
