# model settings
model = dict(
    type='PEM',
    pem_feat_dim=32,
    pem_hidden_dim=256,
    pem_u_ratio_m=1,
    pem_u_ratio_l=2,
    pem_high_temporal_iou_threshold=0.6,
    pem_low_temporal_iou_threshold=2.2,
    soft_nms_alpha=0.75,
    soft_nms_low_threshold=0.65,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
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

# optimizer
optimizer = dict(
    type='Adam', lr=0.01, weight_decay=0.00001)  # this lr is used for 1 gpus

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=10)

total_epochs = 20
checkpoint_config = dict(interval=1, filename_tmpl='pem_epoch_{}.pth')

evaluation = dict(interval=1, metrics=['AR@AN'])

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
