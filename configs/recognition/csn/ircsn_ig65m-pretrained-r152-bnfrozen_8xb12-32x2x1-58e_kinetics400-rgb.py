_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'))

# dataset settings
dataset_type = 'VideoDataset'
root = './data/kinetics400/'
data_root = root + 'videos_train'
data_root_val = root + 'videos_val'
data_root_test = data_root_val

ann_file_train = root + 'kinetics400_train_list_videos.txt'
ann_file_val = root + 'kinetics400_val_list_videos.txt'
ann_file_test = ann_file_val

# file_client_args = dict(
#      io_backend='petrel',
#      path_mapping=dict(
#          {'data/kinetics400': 's3://openmmlab/datasets/action/Kinetics400'}))
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=58, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=16),
    dict(
        type='MultiStepLR',
        begin=0,
        end=58,
        by_epoch=True,
        milestones=[32, 48],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(interval=2, max_keep_ckpts=5))
find_unused_parameters = True

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (12 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=96)
