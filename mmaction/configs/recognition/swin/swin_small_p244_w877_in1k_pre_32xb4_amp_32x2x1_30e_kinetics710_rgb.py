# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .swin_small_p244_w877_in1k_pre_8xb8_amp_32x2x1_30e_kinetics400_rgb import *  # noqa: E501

from mmengine.dataset import DefaultSampler
from torch.utils.data import ConcatDataset

model.update(dict(cls_head=dict(num_classes=710)))

k400_data_root = 'data/kinetics400/videos_train'
k600_data_root = 'data/kinetics600/videos'
k700_data_root = 'data/kinetics700/videos'
k400_data_root_val = 'data/kinetics400/videos_val'
k600_data_root_val = k600_data_root
k700_data_root_val = k700_data_root

k400_ann_file_train = 'data/kinetics710/k400_train_list_videos.txt'
k600_ann_file_train = 'data/kinetics710/k600_train_list_videos.txt'
k700_ann_file_train = 'data/kinetics710/k700_train_list_videos.txt'

k400_ann_file_val = 'data/kinetics710/k400_val_list_videos.txt'
k600_ann_file_val = 'data/kinetics710/k600_val_list_videos.txt'
k700_ann_file_val = 'data/kinetics710/k700_val_list_videos.txt'

k400_trainset = dict(
    type=VideoDataset,
    ann_file=k400_ann_file_train,
    data_prefix=dict(video=k400_data_root),
    pipeline=train_pipeline)
k600_trainset = dict(
    type=VideoDataset,
    ann_file=k600_ann_file_train,
    data_prefix=dict(video=k600_data_root),
    pipeline=train_pipeline)
k700_trainset = dict(
    type=VideoDataset,
    ann_file=k700_ann_file_train,
    data_prefix=dict(video=k700_data_root),
    pipeline=train_pipeline)

k400_valset = dict(
    type=VideoDataset,
    ann_file=k400_ann_file_val,
    data_prefix=dict(video=k400_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)
k600_valset = dict(
    type=VideoDataset,
    ann_file=k600_ann_file_val,
    data_prefix=dict(video=k600_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)
k700_valset = dict(
    type=VideoDataset,
    ann_file=k700_ann_file_val,
    data_prefix=dict(video=k700_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)

k400_testset = k400_valset.copy()
k600_testset = k600_valset.copy()
k700_testset = k700_valset.copy()
k400_testset['pipeline'] = test_pipeline
k600_testset['pipeline'] = test_pipeline
k700_testset['pipeline'] = test_pipeline

k710_trainset = dict(
    type=ConcatDataset,
    datasets=[k400_trainset, k600_trainset, k700_trainset],
    _delete_=True)
k710_valset = dict(
    type=ConcatDataset,
    datasets=[k400_valset, k600_valset, k700_valset],
    _delete_=True)
k710_testset = dict(
    type=ConcatDataset,
    datasets=[k400_testset, k600_testset, k700_testset],
    _delete_=True,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=k710_trainset)
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=k710_valset)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=k710_testset)

optim_wrapper.update(dict(optimizer=dict(lr=2e-3)))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (8 samples per GPU).
auto_scale_lr.update(dict(enable=False, base_batch_size=128))
