_base_ = [
    '../../_base_/models/bmn_400x100.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ImigueDataset'
data_root = 'data/iMiGUE/data/tsn_feature/clip_feature_tsn_depth8_clip_length100_overlap0.5/'
data_root_val = 'data/iMiGUE/data/tsn_feature/clip_feature_tsn_depth8_clip_length100_overlap0.5/'

ann_file_train = 'data/iMiGUE/label/imigue_clip_annotation_100_8_bmn.json'
ann_file_val = 'data/iMiGUE/label/imigue_clip_annotation_100_8_bmn.json'
ann_file_test = 'data/iMiGUE/label/imigue_clip_annotation_100_8_bmn.json'

test_pipeline = [
    dict(type='LoadLocalizationFeatureWithPadding'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature']),
]
train_pipeline = [
    dict(type='LoadLocalizationFeatureWithPadding'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeatureWithPadding'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        subset='testing',
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        subset='training',
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        subset='training',
        data_prefix=data_root))
evaluation = dict(interval=1, metrics=['AR@AN'])

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)
total_epochs = 9

# runtime settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/bsn_imigue_tsn_100_1x8_0.5/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
