# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from ..._base_.models.slowfast_r50 import *
    from ..._base_.default_runtime import *

from mmengine.dataset import DefaultSampler
from mmengine.optim import CosineAnnealingLR, LinearLR
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim import SGD

from mmaction.datasets import (CenterCrop, DecordDecode, DecordInit, Flip,
                               FormatShape, PackActionInputs,
                               RandomResizedCrop, Resize, SampleFrames,
                               ThreeCrop, VideoDataset)
from mmaction.evaluation import AccMetric

data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type=DecordInit, **file_client_args),
    dict(type=SampleFrames, clip_len=32, frame_interval=2, num_clips=1),
    dict(type=DecordDecode),
    dict(type=Resize, scale=(-1, 256)),
    dict(type=RandomResizedCrop),
    dict(type=Resize, scale=(224, 224), keep_ratio=False),
    dict(type=Flip, flip_ratio=0.5),
    dict(type=FormatShape, input_format='NCTHW'),
    dict(type=PackActionInputs)
]
val_pipeline = [
    dict(type=DecordInit, **file_client_args),
    dict(
        type=SampleFrames,
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type=DecordDecode),
    dict(type=Resize, scale=(-1, 256)),
    dict(type=CenterCrop, crop_size=224),
    dict(type=FormatShape, input_format='NCTHW'),
    dict(type=PackActionInputs)
]
test_pipeline = [
    dict(type=DecordInit, **file_client_args),
    dict(
        type=SampleFrames,
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type=DecordDecode),
    dict(type=Resize, scale=(-1, 256)),
    dict(type=ThreeCrop, crop_size=256),
    dict(type=FormatShape, input_format='NCTHW'),
    dict(type=PackActionInputs)
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=VideoDataset,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VideoDataset,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=VideoDataset,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type=AccMetric)
test_evaluator = val_evaluator

train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=256, val_begin=1, val_interval=5)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

optim_wrapper = dict(
    optimizer=dict(type=SGD, lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=34,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=256,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=256)
]

default_hooks.update(
    dict(
        checkpoint=dict(interval=4, max_keep_ckpts=3),
        logger=dict(interval=100)))
