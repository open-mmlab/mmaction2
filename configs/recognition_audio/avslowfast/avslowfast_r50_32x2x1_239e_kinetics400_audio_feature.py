# model settings
model = dict(
    type='AVRecognizer',
    backbone=dict(type='AVResNet3dSlowFast', pretrained=None),
    cls_head=dict(
        type='AVSlowFastHead',
        num_classes=400,
        in_channels=3328,
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'AudioFeatureRawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
audio_prefix = 'data/kinetics400/audio_feature'
ann_file_train = 'data/kinetics400/clean_kinetics400_train_list_audio_feature.txt'  # noqa: E501
ann_file_val = 'data/kinetics400/clean_kinetics400_val_list_audio_feature.txt'
ann_file_test = 'data/kinetics400/clean_kinetics400_val_list_audio_feature.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['imgs', 'audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'audios', 'label'])
]
val_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['imgs', 'audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'audios', 'label'])
]
test_pipeline = [
    dict(type='LoadAudioFeature'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='AudioFeatureSelector'),
    dict(type='FormatAudioShape', input_format='NCTF'),
    dict(type='Collect', keys=['imgs', 'audios', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'audios', 'label'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        audio_prefix=audio_prefix,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        audio_prefix=audio_prefix,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        audio_prefix=audio_prefix,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 239
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('./work_dirs/' +
            'avslowfast_r50_32x2x1_239e_kinetics400_audio_feature/')
load_from = None
resume_from = None
workflow = [('train', 1)]
