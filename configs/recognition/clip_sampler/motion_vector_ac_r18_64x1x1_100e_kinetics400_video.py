# model settings
model = dict(
    type='AudioRecognizer',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='AudioTSNHead',
        num_classes=400,
        in_channels=512,
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_test'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
train_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='PyAVMotionVectorDecode'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['mvs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['mvs', 'label'])
]
val_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='PyAVMotionVectorDecode'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['mvs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['mvs', 'label'])
]
test_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='PyAVMotionVectorDecode'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['i_frames', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['i_frames', 'label'])
]
data = dict(
    videos_per_gpu=8,
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
# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 100
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('./work_dirs/' +
            'motion_vector_ac_r18_64x1x1_100e_kinetics400_video/')
load_from = None
resume_from = None
workflow = [('train', 1)]
