_base_ = ['../../_base_/models/asformer.py', '../../_base_/default_runtime.py']
dataset_type = 'ActionSegmentDataset'
data_root = 'data/gtea/csv_mean_100/'
data_root_val = 'data/action_seg/gtea/'
ann_file_train = 'data/action_seg/gtea/splits/train.split1.bundle'
ann_file_val = 'data/action_seg/gtea/splits/test.split1.bundle'

ann_file_test = 'data/action_seg/gtea/splits/test.split1.bundle'

train_pipeline = [
    dict(type='LoadSegmentationFeature'),
    dict(type='GenerateSegmentationLabels'),
    dict(
        type='PackLocalizationInputs',
        keys=('gt_bbox', ),
        meta_keys=('video_name', ))
]

val_pipeline = [
    dict(type='LoadSegmentationFeature'),
    dict(type='GenerateSegmentationLabels'),
    dict(
        type='PackLocalizationInputs',
        keys=('gt_bbox', ),
        meta_keys=('video_name', 'duration_second', 'duration_frame',
                   'annotations', 'feature_frame'))
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
    batch_size=8,
    num_workers=8,
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

max_epochs = 9
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[
            7,
        ],
        gamma=0.1)
]

work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'
load_from = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/epoch-120.pth'
test_evaluator = dict(
    type='SegmentMetric',
    metric_type='ALL',
    dump_config=dict(out=f'{work_dir}/results.json', output_format='json'))
val_evaluator = test_evaluator
