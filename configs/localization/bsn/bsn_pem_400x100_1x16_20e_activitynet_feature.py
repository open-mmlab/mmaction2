_base_ = [
    '../../_base_/models/bsn_pem.py', '../../_base_/schedules/adam_20e.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
data_root_val = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
ann_file_train = 'data/ActivityNet/anet_anno_train.json'
ann_file_val = 'data/ActivityNet/anet_anno_val.json'
ann_file_test = 'data/ActivityNet/anet_anno_val.json'

work_dir = 'work_dirs/bsn_400x100_20e_1x16_activitynet_feature/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

test_pipeline = [
    dict(
        type='LoadProposals',
        top_k=1000,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['bsp_feature'])
]

train_pipeline = [
    dict(
        type='LoadProposals',
        top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'reference_temporal_iou'],
        meta_name='video_meta',
        meta_keys=[]),
    dict(type='ToTensor', keys=['bsp_feature', 'reference_temporal_iou']),
    dict(
        type='ToDataContainer',
        fields=(dict(key='bsp_feature', stack=False),
                dict(key='reference_temporal_iou', stack=False)))
]

val_pipeline = [
    dict(
        type='LoadProposals',
        top_k=1000,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='Collect',
        keys=['bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['bsp_feature'])
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
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))
evaluation = dict(interval=1, metrics=['AR@AN'])

# runtime settings
checkpoint_config = dict(interval=1, filename_tmpl='pem_epoch_{}.pth')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
