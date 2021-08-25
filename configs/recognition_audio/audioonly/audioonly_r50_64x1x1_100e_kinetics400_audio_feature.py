_base_ = [
    '../../_base_/models/audioonly_r50.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'AudioFeatureDataset'
data_root = 'data/kinetics400/audio_feature_train'
data_root_val = 'data/kinetics400/audio_feature_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_audio_feature.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_audio_feature.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_audio_feature.txt'
train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
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
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
test_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['audios'])
]
data = dict(
    videos_per_gpu=160,
    workers_per_gpu=2,
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
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=2.0, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(interval=1)
work_dir = ('./work_dirs/' +
            'audioonly_r50_64x1x1_100e_kinetics400_audio_feature/')
