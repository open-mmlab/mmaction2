# Finetuning Models

This tutorial provides instructions for users to use the pre-trained models
to finetune them on other datasets, so that better performance can be achieved.

- [Finetuning Models](#finetuning-models)
  - [Outline](#outline)
  - [Choose Template Config](#choose-template-config)
  - [Modify Head](#modify-head)
  - [Modify Dataset](#modify-dataset)
  - [Modify Training Schedule](#modify-training-schedule)
  - [Use Pre-Trained Model](#use-pre-trained-model)
  - [Start Training](#start-training)

## Outline

There are two steps to finetune a model on a new dataset.

1. Add support for the new dataset. See [Prepare Dataset](prepare_dataset.md) and [Customize Dataset](../advanced_guides/customize_dataset.md).
2. Modify the configs. This will be discussed in this tutorial.

## Choose Template Config

Here, we would like to take `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` as an example. We first copy this config file to the same folder and rename it to `tsn_ucf101.py`, then four parts in the config need attention, specifically, add new keys for non-existing keys and modify the original keys for existing keys.

## Modify Head

The `num_classes` in the `cls_head` need to be changed to the class number of the new dataset.
The weights of the pre-trained models are reused except for the final prediction layer.
So it is safe to change the class number.
In our case, UCF101 has 101 classes.
So we change it from 400 (class number of Kinetics-400) to 101.

```python
# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=101  # change from 400 to 101
        ))
```

## Modify Dataset

MMAction2 supports UCF101, Kinetics-400, Moments in Time, Multi-Moments in Time, THUMOS14,
Something-Something V1&V2, ActivityNet Dataset.
The users may need to adapt one of the above datasets to fit their special datasets.
You could refer to [Prepare Dataset](prepare_dataset.md) and [Customize Dataset](../advanced_guides/customize_dataset.md) for more details.
In our case, UCF101 is already supported by various dataset types, like `VideoDataset`,
so we change the config as follows.

```python
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos_train/'
data_root_val = 'data/ucf101/videos_val/'
ann_file_train = 'data/ucf101/ucf101_train_list.txt'
ann_file_val = 'data/ucf101/ucf101_val_list.txt'
```

## Modify Training Schedule

Finetuning usually requires a smaller learning rate and fewer training epochs.

```python
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # change from 0.01 to 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))
```

## Use Pre-Trained Model

To use the pre-trained model for the whole network, the new config adds the link of pre-trained models in the `load_from`.
We set `load_from=None` as default in `configs/_base_/default_runtime.py` and owing to [inheritance design](config.md), users can directly change it by setting `load_from` in their configs.

```python
# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'  # model path can be found in model zoo
```

## Start Training

Now, we have finished the fine-tuning config file as follows:

```python
_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=101  # change from 400 to 101
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
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # change from 0.01 to 0.005
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

An easier way is to inherit the kinetics400 config and only specify the modified keys. Please make sure that the custom config is in the same folder with `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py`.

```python
_base_ = [
    'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'  # inherit template config
]

# model settings
model = dict(
    cls_head=dict(
        type='TSNHead',
        num_classes=101))  # change from 400 to 101


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
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005, # change from 0.01 to 0.005
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# use the pre-trained model for the whole TSN network
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'

```

You can use the following command to finetune a model on your dataset.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the TSN model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsn/tsn_ucf101.py  \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](train_test.md).
