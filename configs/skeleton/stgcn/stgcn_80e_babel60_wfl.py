samples_per_cls = [
    518, 1993, 6260, 508, 208, 3006, 431, 724, 4527, 2131, 199, 1255, 487, 302,
    136, 571, 267, 646, 1180, 405, 731, 842, 1619, 271, 1198, 1012, 865, 462,
    526, 405, 487, 168, 271, 609, 503, 167, 415, 421, 283, 2069, 715, 196, 989,
    122, 599, 396, 245, 380, 236, 260, 325, 133, 206, 191, 394, 145, 277, 268,
    172, 146
]

model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        num_person=1,
        loss_cls=dict(type='CBFocalLoss', samples_per_cls=samples_per_cls)),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = 'data/babel/babel60_train.pkl'
ann_file_val = 'data/babel/babel60_val.pkl'
train_pipeline = [
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix='',
            pipeline=train_pipeline)),
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
lr_config = dict(policy='step', step=[10, 14])
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stgcn_80e_babel60_wfl/'
load_from = None
resume_from = None
workflow = [('train', 1)]
