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

train_pipeline = [
    dict(
        type='LoadProposals',
        top_k=500,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='PackLocalizationInputs',
        keys=('reference_temporal_iou', 'bsp_feature'),
        meta_keys=())
]
val_pipeline = [
    dict(
        type='LoadProposals',
        top_k=1000,
        pgm_proposals_dir=pgm_proposals_dir,
        pgm_features_dir=pgm_features_dir),
    dict(
        type='PackLocalizationInputs',
        keys=('tmin', 'tmax', 'tmin_score', 'tmax_score', 'bsp_feature'),
        meta_keys=('video_name', 'duration_second', 'duration_frame',
                   'annotations', 'feature_frame')),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

train_cfg = dict(val_interval=20)

test_evaluator = dict(
    type='ANetMetric',
    metric_type='AR@AN',
    dump_config=dict(out=f'{work_dir}/results.json', output_format='json'))
val_evaluator = test_evaluator
