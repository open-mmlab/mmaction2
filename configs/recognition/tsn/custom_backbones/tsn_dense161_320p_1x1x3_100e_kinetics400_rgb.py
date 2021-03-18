_base_ = [
    '../../../_base_/schedules/sgd_100e.py',
    '../../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(type='torchvision.densenet161', pretrained=True),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=2208,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train_320p'
data_root_val = 'data/kinetics400/rawframes_val_320p'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes_320p.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes_320p.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes_320p.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
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
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# runtime settings
work_dir = './work_dirs/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/'
optimizer = dict(
    type='SGD',
    lr=0.00375,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
