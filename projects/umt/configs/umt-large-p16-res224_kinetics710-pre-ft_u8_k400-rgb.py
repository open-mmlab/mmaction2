custom_imports = dict(imports='models')

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UMTViT',
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        all_frames=8,
        qkv_bias=True),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=400,
        in_channels=1024,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root_val = 'data/kinetics400/videos_val'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(io_backend='disk')

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=8, num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='AccMetric')
test_cfg = dict(type='TestLoop')

default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
