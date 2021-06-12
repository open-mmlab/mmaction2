_base_ = ['../../_base_/models/tsn_r50.py', '../../_base_/default_runtime.py']

# model settings
# ``in_channels`` should be 2 * clip_len
model = dict(
    backbone=dict(in_channels=10),
    cls_head=dict(num_classes=200, dropout_ratio=0.8))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ActivityNet/rawframes'
data_root_val = 'data/ActivityNet/rawframes'
ann_file_train = 'data/ActivityNet/anet_train_video.txt'
ann_file_val = 'data/ActivityNet/anet_val_video.txt'
ann_file_test = 'data/ActivityNet/anet_val_clip.txt'
img_norm_cfg = dict(mean=[128, 128], std=[128, 128], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=5, frame_interval=1, num_clips=8),
    dict(type='FrameSelector'),
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
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
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
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW_Flow'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        start_index=0,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='flow_{}_{:05d}.jpg',
        modality='Flow',
        start_index=0,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='flow_{}_{:05d}.jpg',
        with_offset=True,
        modality='Flow',
        start_index=0,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[60, 120])
total_epochs = 150

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/'
load_from = ('https://download.openmmlab.com/mmaction/recognition/tsn/'
             'tsn_r50_320p_1x1x8_110e_kinetics400_flow/'
             'tsn_r50_320p_1x1x8_110e_kinetics400_flow_20200705-1f39486b.pth')
workflow = [('train', 5)]
