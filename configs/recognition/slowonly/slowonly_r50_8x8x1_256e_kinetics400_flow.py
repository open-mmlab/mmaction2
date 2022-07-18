_base_ = [
    '../../_base_/models/slowonly_r50.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(in_channels=2, with_pool2=False))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
ann_file_train = 'data/kinetics400/kinetics_flow_train_list.txt'
ann_file_val = 'data/kinetics400/kinetics_flow_val_list.txt'
ann_file_test = 'data/kinetics400/kinetics_flow_val_list.txt'
img_norm_cfg = dict(mean=[128, 128], std=[128, 128])
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        modality='Flow',
        filename_tmpl='{}_{:05d}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        modality='Flow',
        filename_tmpl='{}_{:05d}.jpg',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        modality='Flow',
        filename_tmpl='{}_{:05d}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

val_cfg = dict(interval=5)
test_cfg = dict()
# optimizer
optimizer = dict(
    type='SGD', lr=0.06, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus

# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=256,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=256)
]

total_epochs = 196
train_cfg = dict(by_epoch=True, max_epochs=196)
# runtime settings

default_hooks = dict(
    optimizer=dict(grad_clip=dict(max_norm=40, norm_type=2)),
    checkpoint=dict(interval=4))
