_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(fusion_kernel=7)),
    cls_head=dict(num_classes=174))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/sthv1/rawframes'
data_root_val = 'data/sthv1/rawframes'
ann_file_train = 'data/sthv1/sthv1_train_list_rawframes.txt'
ann_file_val = 'data/sthv1/sthv1_val_list_rawframes.txt'
ann_file_test = 'data/sthv1/sthv1_val_list_rawframes.txt'

file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict({'data/sthv1': 's3://openmmlab/datasets/action/sthv1'}))
# file_client_args = dict(io_backend='disk')
sthv1_flip_label_map = {2: 4, 4: 2, 30: 41, 41: 30, 52: 66, 66: 52}
train_pipeline = [
    dict(type='SampleFrames', clip_len=64, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv1_flip_label_map),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        filename_tmpl='{:05}.jpg',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        filename_tmpl='{:05}.jpg',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=22, val_begin=18, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=1e-06),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=510),
    dict(
        type='MultiStepLR',
        begin=0,
        end=22,
        by_epoch=True,
        milestones=[14, 18],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=3), logger=dict(interval=100))

load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth'  # noqa: E501
