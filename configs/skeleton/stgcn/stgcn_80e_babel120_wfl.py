samples_per_cls = [
    518, 1993, 6260, 508, 208, 3006, 431, 724, 4527, 2131, 199, 1255, 487, 302,
    136, 571, 267, 646, 1180, 405, 72, 731, 842, 1619, 271, 27, 1198, 1012,
    110, 865, 462, 526, 405, 487, 101, 24, 84, 64, 168, 271, 609, 503, 76, 167,
    415, 137, 421, 283, 2069, 715, 196, 66, 44, 989, 122, 43, 599, 396, 245,
    380, 34, 236, 260, 325, 127, 133, 119, 66, 125, 50, 206, 191, 394, 69, 98,
    145, 38, 21, 29, 64, 277, 65, 39, 31, 35, 85, 54, 80, 133, 66, 39, 64, 268,
    34, 172, 54, 33, 21, 110, 19, 40, 55, 146, 39, 37, 75, 101, 20, 46, 55, 43,
    21, 43, 87, 29, 36, 24, 37, 28, 39
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
        num_classes=120,
        in_channels=256,
        num_person=1,
        loss_cls=dict(type='CBFocalLoss', samples_per_cls=samples_per_cls)),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = 'data/babel/babel120_train.pkl'
ann_file_val = 'data/babel/babel120_val.pkl'
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
work_dir = './work_dirs/stgcn_80e_babel120_wfl/'
load_from = None
resume_from = None
workflow = [('train', 1)]
