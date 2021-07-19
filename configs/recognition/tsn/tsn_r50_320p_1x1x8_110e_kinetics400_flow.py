_base_ = ['../../_base_/models/tsn_r50.py', '../../_base_/default_runtime.py']

# model settings
# ``in_channels`` should be 2 * clip_len
model = dict(backbone=dict(in_channels=10))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train_320p'
data_root_val = 'data/kinetics400/rawframes_val_320p'
ann_file_train = 'data/kinetics400/kinetics400_flow_train_list_320p.txt'
ann_file_val = 'data/kinetics400/kinetics400_flow_val_list_320p.txt'
ann_file_test = 'data/kinetics400/kinetics400_flow_val_list_320p.txt'
img_norm_cfg = dict(mean=[128, 128], std=[128, 128])
train_pipeline = [
    dict(type='SampleFrames', clip_len=5, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=5,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{}_{:05d}.jpg',
        modality='Flow',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{}_{:05d}.jpg',
        modality='Flow',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{}_{:05d}.jpg',
        modality='Flow',
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.001875, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[70, 100])
total_epochs = 110

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r50_320p_1x1x8_110e_kinetics400_flow/'
