_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='RecognizerOmni',
    backbone=dict(type='OmniResNet'),
    cls_head=dict(
        type='OmniHead',
        image_classes=1000,
        video_classes=400,
        in_channels=2048))

# dataset settings
image_root = 'data/imagenet'
image_ann_train = 'data/imagenet/train.txt'

video_root = 'data/kinetics400/videos_train'
video_root_val = 'data/kinetics400/videos_val'
video_ann_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
video_ann_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

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
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VideoDataset',
        ann_file=video_ann_train,
        data_prefix=dict(video=video_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=video_ann_val,
        data_prefix=dict(video=video_ann_val),
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
        data_prefix=dict(video=video_ann_val),
        pipeline=test_pipeline,
        test_mode=True))

imagenet_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(type='mmcls.RandomResizedCrop', scale=224),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs'),
]

image_dataloader = dict(
    batch_size=80,
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
    max_epochs=150)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=10,
        end=150,
        by_epoch=True,
        milestones=[90, 130],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))
