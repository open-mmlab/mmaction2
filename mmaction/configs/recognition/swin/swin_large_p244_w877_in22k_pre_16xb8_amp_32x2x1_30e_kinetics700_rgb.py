# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .swin_large_p244_w877_in22k_pre_8xb8_amp_32x2x1_30e_kinetics400_rgb import *  # noqa: E501

model.update(dict(cls_head=dict(num_classes=700)))

# dataset
data_root = 'data/kinetics700/videos_train'
data_root_val = 'data/kinetics700/videos_val'
ann_file_train = 'data/kinetics700/kinetics700_train_list_videos.txt'
ann_file_val = 'data/kinetics700/kinetics700_val_list_videos.txt'
ann_file_test = 'data/kinetics700/kinetics700_val_list_videos.txt'

dataset_type = VideoDataset
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
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
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

optim_wrapper.update(dict(optimizer=dict(lr=2e-3)))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (8 samples per GPU).
auto_scale_lr.update(dict(enable=False, base_batch_size=128))
