_base_ = [
    '../../../_base_/models/tsn_r50.py',
    '../../../_base_/schedules/sgd_100e.py',
    '../../../_base_/default_runtime.py'
]

# model settings
category_nums = dict(
    action=739, attribute=117, concept=291, event=69, object=1678, scene=248)
target_cate = 'scene'

model = dict(
    backbone=dict(pretrained='torchvision://resnet18', depth=18),
    cls_head=dict(
        in_channels=512,
        num_classes=category_nums[target_cate],
        multi_class=True,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=333.)))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/hvu/rawframes_train'
data_root_val = 'data/hvu/rawframes_val'
ann_file_train = f'data/hvu/hvu_{target_cate}_train.json'
ann_file_val = f'data/hvu/hvu_{target_cate}_val.json'
ann_file_test = f'data/hvu/hvu_{target_cate}_val.json'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=category_nums[target_cate],
        filename_tmpl='img_{:05d}.jpg'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=True,
        num_classes=category_nums[target_cate],
        filename_tmpl='img_{:05d}.jpg'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=True,
        num_classes=category_nums[target_cate],
        filename_tmpl='img_{:05d}.jpg'))
evaluation = dict(interval=2, metrics=['mean_average_precision'])

# runtime settings
work_dir = f'./work_dirs/tsn_r18_1x1x8_100e_hvu_{target_cate}_rgb/'
