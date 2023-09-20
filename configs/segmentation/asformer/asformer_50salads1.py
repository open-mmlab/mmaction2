_base_ = [
    '../../_base_/models/asformer.py', '../../_base_/default_runtime.py'
]  # dataset settings
dataset_type = 'ActionSegmentDataset'
data_root = 'data/action_seg/50salads/'
data_root_val = 'data/action_seg/50salads/'
ann_file_train = 'data/action_seg/50salads/splits/train.split1.bundle'
ann_file_val = 'data/action_seg/50salads/splits/test.split1.bundle'
ann_file_test = 'data/action_seg/50salads/splits/test.split1.bundle'

model = dict(
    type='ASFormer',
    num_layers=10,
    num_f_maps=64,
    input_dim=2048,
    num_decoders=3,
    num_classes=19,
    channel_masking_rate=0.3,
    sample_rate=2,
    r1=2,
    r2=2)

train_pipeline = [
    dict(type='LoadSegmentationFeature'),
    dict(
        type='PackSegmentationInputs',
        keys=('classes', ),
        meta_keys=(
            'num_classes',
            'actions_dict',
            'index2label',
            'ground_truth',
            'classes',
        ))
]

val_pipeline = [
    dict(type='LoadSegmentationFeature'),
    dict(
        type='PackSegmentationInputs',
        keys=('classes', ),
        meta_keys=('num_classes', 'actions_dict', 'index2label',
                   'ground_truth', 'classes'))
]

test_pipeline = [
    dict(type='LoadSegmentationFeature'),
    dict(
        type='PackSegmentationInputs',
        keys=('classes', ),
        meta_keys=('num_classes', 'actions_dict', 'index2label',
                   'ground_truth', 'classes'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
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

max_epochs = 120
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=0,
    val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005, weight_decay=1e-5))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[
            80,
            100,
        ],
        gamma=0.5)
]
work_dir = './work_dirs/50salads1/'
test_evaluator = dict(
    type='SegmentMetric',
    metric_type='ALL',
    dump_config=dict(out=f'{work_dir}/results.json', output_format='json'))
val_evaluator = test_evaluator
default_hooks = dict(checkpoint=dict(interval=10, max_keep_ckpts=3))
