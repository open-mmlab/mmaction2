_base_ = [
    '../../_base_/models/stgcn.py',
    '../../_base_/default_runtime.py'
]

model = dict(backbone=dict(graph_cfg=dict(layout='coco', strategy='spatial')))

dataset_type = 'PoseDataset'
ann_file_train = 'data/ntu60/xsub/train_2d.pkl'
ann_file_val = 'data/ntu60/xsub/val_2d.pkl'
keypoint_norm_cfg = dict(
    mean=[960, 540, 0.5], min_value=[0., 0., 0.], max_value=[1920, 1080, 1.])
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize', **keypoint_norm_cfg),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=300),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalize', **keypoint_norm_cfg),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_train, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_val, pipeline=test_pipeline, test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, ann_file=ann_file_val, pipeline=test_pipeline, test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=80)
val_cfg = dict(interval=1)
test_cfg = dict()

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[10, 50],
        gamma=0.1)
]

optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)