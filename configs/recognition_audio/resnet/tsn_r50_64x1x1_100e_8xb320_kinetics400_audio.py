_base_ = [
    '../../_base_/models/tsn_r50_audio.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'AudioDataset'
data_root = 'data/kinetics400/audios_train'
data_root_val = 'data/kinetics400/audios_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_audios.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_audios.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_audios.txt'
train_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecode'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='AudioDecodeInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='AudioDecodeInit'),
    dict(type='AudioAmplify', ratio=1.5),
    dict(type='MelSpectrogram'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=320,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(audio=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=320,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=dict(audio=data_root_val),
        test_mode=True))
test_dataloader = dict(
    batch_size=320,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=dict(audio=data_root_val),
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=100,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, interval=5),
                     logger=dict(interval=1))
