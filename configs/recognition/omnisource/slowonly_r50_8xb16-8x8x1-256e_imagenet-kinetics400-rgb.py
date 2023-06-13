_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='RecognizerOmni',
    backbone=dict(type='OmniResNet'),
    cls_head=dict(
        type='OmniHead',
        image_classes=1000,
        video_classes=400,
        in_channels=2048,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='MIX2d3d'))

# dataset settings
image_root = 'data/imagenet/'
image_ann_train = 'meta/train.txt'

video_root = 'data/kinetics400/videos_train'
video_root_val = 'data/kinetics400/videos_val'
video_ann_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
video_ann_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

num_images = 1281167  # number of training samples in the ImageNet dataset
num_videos = 240435  # number of training samples in the Kinetics400 dataset
batchsize_video = 16
num_gpus = 8
num_iter = num_videos // (batchsize_video * num_gpus)
batchsize_image = num_images // (num_iter * num_gpus)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=batchsize_video,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VideoDataset',
        ann_file=video_ann_train,
        data_prefix=dict(video=video_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=video_ann_val,
        data_prefix=dict(video=video_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=video_ann_val,
        data_prefix=dict(video=video_root_val),
        pipeline=test_pipeline,
        test_mode=True))

imagenet_pipeline = [
    dict(type='LoadRGBFromFile'),
    dict(type='mmcls.RandomResizedCrop', scale=224),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs'),
]

image_dataloader = dict(
    batch_size=batchsize_image,
    num_workers=8,
    dataset=dict(
        type='mmcls.ImageNet',
        data_root=image_root,
        ann_file=image_ann_train,
        data_prefix='train',
        pipeline=imagenet_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='MultiLoaderEpochBasedTrainLoop',
    other_loaders=[image_dataloader],
    max_epochs=256,
    val_interval=4)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=34,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=222,
        eta_min=0,
        by_epoch=True,
        begin=34,
        end=256,
        convert_to_iter_based=True)
]
"""
The learning rate is for total_batch_size = 8 x 16 (num_gpus x batch_size)
If you want to use other batch size or number of GPU settings, please update
the learning rate with the linear scaling rule.
"""
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# runtime settings
default_hooks = dict(checkpoint=dict(interval=4, max_keep_ckpts=3))
