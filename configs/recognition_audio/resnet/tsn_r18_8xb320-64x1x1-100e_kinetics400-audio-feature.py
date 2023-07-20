_base_ = '../../_base_/default_runtime.py'

# model settings
model = dict(
    type='RecognizerAudio',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='TSNAudioHead',
        num_classes=400,
        in_channels=512,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'))

# dataset settings
dataset_type = 'AudioDataset'
data_root = 'data/kinetics400'
ann_file_train = 'kinetics400_train_list_audio_features.txt'
ann_file_val = 'kinetics400_val_list_audio_features.txt'
ann_file_test = 'kinetics400_val_list_audio_features.txt'

train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=320,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_root=data_root,
        data_prefix=dict(audio='audio_features_train')))
val_dataloader = dict(
    batch_size=320,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_root=data_root,
        data_prefix=dict(audio='audio_features_val'),
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_root=data_root,
        data_prefix=dict(audio='audio_features_val'),
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=0, T_max=100, by_epoch=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, interval=5))
