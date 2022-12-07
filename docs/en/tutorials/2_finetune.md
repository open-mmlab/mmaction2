# Tutorial 2: Finetuning Models

This tutorial provides instructions for users to use the pre-trained models
to finetune them on other datasets, so that better performance can be achieved.

<!-- TOC -->

- [Tutorial 2: Finetuning Models](#tutorial-2-finetuning-models)
  - [Outline](#outline)
  - [Modify Head](#modify-head)
  - [Modify Dataset](#modify-dataset)
  - [Modify Training Schedule](#modify-training-schedule)
  - [Use Pre-Trained Model](#use-pre-trained-model)

<!-- TOC -->

## Outline

There are two steps to finetune a model on a new dataset.

1. Add support for the new dataset. See [Tutorial 3: Adding New Dataset](3_new_dataset.md).
2. Modify the configs. This will be discussed in this tutorial.

For example, if the users want to finetune models pre-trained on Kinetics-400 Dataset to another dataset, say UCF101,
then four parts in the config (see [here](1_config.md)) needs attention.

## Modify Head

The `num_classes` in the `cls_head` need to be changed to the class number of the new dataset.
The weights of the pre-trained models are reused except for the final prediction layer.
So it is safe to change the class number.
In our case, UCF101 has 101 classes.
So we change it from 400 (class number of Kinetics-400) to 101.

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
        num_classes=101,   # change from 400 to 101
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips=None))
```

Note that the `pretrained='torchvision://resnet50'` setting is used for initializing backbone.
If you are training a new model from ImageNet-pretrained weights, this is for you.
However, this setting is not related to our task at hand.
What we need is `load_from`, which will be discussed later.

## Modify Dataset

MMAction2 supports UCF101, Kinetics-400, Moments in Time, Multi-Moments in Time, THUMOS14,
Something-Something V1&V2, ActivityNet Dataset.
The users may need to adapt one of the above dataset to fit for their special datasets.
In our case, UCF101 is already supported by various dataset types, like `RawframeDataset`,
so we change the config as follows.

```python
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ucf101/rawframes_train/'
data_root_val = 'data/ucf101/rawframes_val/'
ann_file_train = 'data/ucf101/ucf101_train_list.txt'
ann_file_val = 'data/ucf101/ucf101_val_list.txt'
ann_file_test = 'data/ucf101/ucf101_val_list.txt'
```

## Modify Training Schedule

Finetuning usually requires smaller learning rate and less training epochs.

```python
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)  # change from 0.01 to 0.005
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50 # change from 100 to 50
checkpoint_config = dict(interval=5)
```

## Use Pre-Trained Model

To use the pre-trained model for the whole network, the new config adds the link of pre-trained models in the `load_from`.
We set `load_from=None` as default in `configs/_base_/default_runtime.py` and owing to \[inheritance design\](/docs/en/tutorials/1_config.md), users can directly change it by setting `load_from` in their configs.

```python
# use the pre-trained model for the whole TSN network
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'  # model path can be found in model zoo
```
