_base_ = ['../../_base_/default_runtime.py']

# model settings
num_frames = 8
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=True,
        pretrained='ViT-B/16'),
    cls_head=dict(
        type='TimeSformerHead',
        dropout_ratio=0.5,
        num_classes=710,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=4,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# dataset settings
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
    type='VideoDataset',
    ann_file=k400_ann_file_train,
    data_prefix=dict(video=k400_data_root),
    pipeline=train_pipeline)
k600_trainset = dict(
    type='VideoDataset',
    ann_file=k600_ann_file_train,
    data_prefix=dict(video=k600_data_root),
    pipeline=train_pipeline)
k700_trainset = dict(
    type='VideoDataset',
    ann_file=k700_ann_file_train,
    data_prefix=dict(video=k700_data_root),
    pipeline=train_pipeline)

k400_valset = dict(
    type='VideoDataset',
    ann_file=k400_ann_file_val,
    data_prefix=dict(video=k400_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)
k600_valset = dict(
    type='VideoDataset',
    ann_file=k600_ann_file_val,
    data_prefix=dict(video=k600_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)
k700_valset = dict(
    type='VideoDataset',
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
    type='ConcatDataset',
    datasets=[k400_trainset, k600_trainset, k700_trainset])
k710_valset = dict(
    type='ConcatDataset', datasets=[k400_valset, k600_valset, k700_valset])
k710_testset = dict(
    type='ConcatDataset',
    datasets=[k400_testset, k600_testset, k700_testset],
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=k710_trainset)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=k710_valset)
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=k710_testset)

val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=55, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min_ratio=0.5,
        by_epoch=True,
        begin=5,
        end=55,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)
