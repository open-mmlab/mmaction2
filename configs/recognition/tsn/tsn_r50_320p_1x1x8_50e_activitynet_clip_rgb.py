_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_50e.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(cls_head=dict(num_classes=200, dropout_ratio=0.8))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ActivityNet/rawframes'
data_root_val = 'data/ActivityNet/rawframes'
ann_file_train = 'data/ActivityNet/anet_train_clip.txt'
ann_file_val = 'data/ActivityNet/anet_val_clip.txt'
ann_file_test = 'data/ActivityNet/anet_val_clip.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        with_offset=True,
        start_index=0,
        filename_tmpl='image_{:05d}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        with_offset=True,
        start_index=0,
        filename_tmpl='image_{:05d}.jpg',
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        with_offset=True,
        start_index=0,
        filename_tmpl='image_{:05d}.jpg',
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

val_cfg = dict(interval=5)
test_cfg = dict()

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

load_from = ('https://download.openmmlab.com/mmaction/recognition/tsn/'
             'tsn_r50_320p_1x1x8_100e_kinetics400_rgb/'
             'tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth')
