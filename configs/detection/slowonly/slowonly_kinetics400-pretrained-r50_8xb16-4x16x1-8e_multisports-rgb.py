_base_ = [
    '../../_base_/default_runtime.py',
]
url = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
       'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-'
       'rgb/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_'
       'kinetics400-rgb_20220901-e7b65fad.pth')
num_classes = 66
model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        pretrained2d=False,
        lateral=False,
        num_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=(1, 2, 2, 1)),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=False,
            in_channels=2048,
            num_classes=num_classes,
            multilabel=False,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
data_root = 'data/multisports/trainval'
anno_root = 'data/multisports/annotations'

ann_file_train = f'{anno_root}/multisports_train.csv'
ann_file_val = f'{anno_root}/multisports_val.csv'
gt_file = f'{anno_root}/multisports_GT.pkl'

proposal_file_train = f'{anno_root}/multisports_dense_proposals_train.recall_96.13.pkl'  # noqa: E501
proposal_file_val = f'{anno_root}/multisports_dense_proposals_val.recall_96.13.pkl'  # noqa: E501

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=16),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleAVAFrames', clip_len=4, frame_interval=16, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        num_classes=num_classes,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
        timestamp_start=1,
        start_index=0,
        use_frames=False,
        fps=1,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        num_classes=num_classes,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True,
        timestamp_start=1,
        start_index=0,
        use_frames=False,
        fps=1,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(type='MultiSportsMetric', ann_file=gt_file)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=8, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[6, 7],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=5, norm_type=2))
