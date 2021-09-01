model = dict(
    type='ST_GCN_18',
    in_channels=3,
    num_class=60,
    dropout=0.5,
    edge_importance_weighting=True,
    # graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial')
    graph_cfg=dict(layout='coco', strategy='spatial'))

dataset_type = 'PoseDataset'
ann_file_train = 'data/posec3d/ntu60_xsub_train.pkl'
ann_file_val = 'data/posec3d/ntu60_xsub_val.pkl'
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=300),
    dict(type='PoseDecode'),
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='FormatNtuPose', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=300, test_mode=True),
    dict(type='PoseDecode'),
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='FormatNtuPose', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=300, test_mode=True),
    dict(type='PoseDecode'),
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='FormatNtuPose', input_format='NCTVM'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 50])
total_epochs = 80
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/st-gcn_ntu-xsub/'
load_from = None
resume_from = None
workflow = [('train', 5), ('val', 1)]
