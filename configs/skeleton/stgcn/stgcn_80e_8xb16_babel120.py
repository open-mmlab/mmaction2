_base_ = ['../../_base_/models/stgcn.py', '../../_base_/default_runtime.py']

model = dict(cls_head=dict(num_classes=120, num_person=1))

dataset_type = 'PoseDataset'
ann_file_train = 'data/babel/babel120_train.pkl'
ann_file_val = 'data/babel/babel120_val.pkl'
train_pipeline = [
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=16, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=16,
        by_epoch=True,
        milestones=[10, 14],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True))

default_hooks = dict(logger=dict(interval=100))
