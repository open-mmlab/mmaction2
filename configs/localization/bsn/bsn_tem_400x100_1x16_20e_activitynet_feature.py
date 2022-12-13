_base_ = ['../../_base_/models/bsn_tem.py', '../../_base_/default_runtime.py']

# dataset settings
dataset_type = 'ImigueDataset'
data_root = 'data/iMiGUE/data/tsn_feature/clip_feature_tsn_depth8_clip_length100_overlap0.5/'
data_root_val = 'data/iMiGUE/data/tsn_feature/clip_feature_tsn_depth8_clip_length100_overlap0.5/'

ann_file_train = 'data/iMiGUE/label/imigue_clip_annotation_100_8_bsn.json'
ann_file_val = 'data/iMiGUE/label/imigue_clip_annotation_100_8_bsn.json'
ann_file_test = 'data/iMiGUE/label/imigue_clip_annotation_100_8_bsn.json'

test_pipeline = [
    dict(type='LoadLocalizationFeatureWithPadding'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature'])
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
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeatureWithPadding'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(type='ToDataContainer', fields=[dict(key='gt_bbox', stack=False)])
]

data = dict(
    videos_per_gpu=16,
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

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)
total_epochs = 20

# runtime settings
checkpoint_config = dict(interval=1, filename_tmpl='tem_epoch_{}.pth')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 1), ('val', 1)]
work_dir = 'work_dirs/bsn_imigue_tsn_100_1x8_0.5'
tem_results_dir = f'{work_dir}/tem_results/'
output_config = dict(out=tem_results_dir, output_format='csv')
