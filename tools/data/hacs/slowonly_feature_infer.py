# Copyright (c) OpenMMLab. All rights reserved.
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=700,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips=None),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

data_root = './data'
ann_file = 'hacs_data.txt'

test_pipeline = [
    dict(type='DecordInit', io_backend='disk'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=100,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file=ann_file,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='DumpResults', out_file_path='result.pkl')
test_cfg = dict(type='TestLoop')
