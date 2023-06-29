# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from ..._base_.default_runtime import *

from mmengine.dataset import DefaultSampler, RepeatDataset
from mmengine.optim import CosineAnnealingLR
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim import SGD

from mmaction.datasets import (CenterCrop, Flip, FormatShape,
                               GeneratePoseTarget, PackActionInputs,
                               PoseCompact, PoseDataset, PoseDecode,
                               RandomResizedCrop, Resize, UniformSampleFrames)
from mmaction.evaluation import AccMetric
from mmaction.models import I3DHead, Recognizer3D, ResNet3dSlowOnly

model = dict(
    type=Recognizer3D,
    backbone=dict(
        type=ResNet3dSlowOnly,
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type=I3DHead,
        in_channels=512,
        num_classes=60,
        dropout_ratio=0.5,
        average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file = 'data/skeleton/ntu60_2d.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
             [1, 3], [2, 4], [11, 12]]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
train_pipeline = [
    dict(type=UniformSampleFrames, clip_len=48),
    dict(type=PoseDecode),
    dict(type=PoseCompact, hw_ratio=1., allow_imgpad=True),
    dict(type=Resize, scale=(-1, 64)),
    dict(type=RandomResizedCrop, area_range=(0.56, 1.0)),
    dict(type=Resize, scale=(56, 56), keep_ratio=False),
    dict(type=Flip, flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type=GeneratePoseTarget,
        sigma=0.6,
        use_score=True,
        with_kp=False,
        with_limb=True,
        skeletons=skeletons),
    dict(type=FormatShape, input_format='NCTHW_Heatmap'),
    dict(type=PackActionInputs)
]
val_pipeline = [
    dict(type=UniformSampleFrames, clip_len=48, num_clips=1, test_mode=True),
    dict(type=PoseDecode),
    dict(type=PoseCompact, hw_ratio=1., allow_imgpad=True),
    dict(type=Resize, scale=(-1, 64)),
    dict(type=CenterCrop, crop_size=64),
    dict(
        type=GeneratePoseTarget,
        sigma=0.6,
        use_score=True,
        with_kp=False,
        with_limb=True,
        skeletons=skeletons),
    dict(type=FormatShape, input_format='NCTHW_Heatmap'),
    dict(type=PackActionInputs)
]
test_pipeline = [
    dict(type=UniformSampleFrames, clip_len=48, num_clips=10, test_mode=True),
    dict(type=PoseDecode),
    dict(type=PoseCompact, hw_ratio=1., allow_imgpad=True),
    dict(type=Resize, scale=(-1, 64)),
    dict(type=CenterCrop, crop_size=64),
    dict(
        type=GeneratePoseTarget,
        sigma=0.6,
        use_score=True,
        with_kp=False,
        with_limb=True,
        skeletons=skeletons,
        double=True,
        left_limb=left_limb,
        right_limb=right_limb),
    dict(type=FormatShape, input_format='NCTHW_Heatmap'),
    dict(type=PackActionInputs)
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=RepeatDataset,
        times=10,
        dataset=dict(
            type=PoseDataset,
            ann_file=ann_file,
            split='xsub_train',
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=PoseDataset,
        ann_file=ann_file,
        split='xsub_val',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=PoseDataset,
        ann_file=ann_file,
        split='xsub_val',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = [dict(type=AccMetric)]
test_evaluator = val_evaluator

train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=24, val_begin=1, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

param_scheduler = [
    dict(
        type=CosineAnnealingLR,
        eta_min=0,
        T_max=24,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type=SGD, lr=0.2, momentum=0.9, weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2))
