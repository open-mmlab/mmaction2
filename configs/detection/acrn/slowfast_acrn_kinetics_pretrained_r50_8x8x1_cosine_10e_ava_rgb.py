_base_ = '../../_base_/default_runtime.py'

model = dict(
    type='mmdet.FastRCNN',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            temporal_pool_mode='max'),
        shared_head=dict(type='ACRNHead', in_channels=4608, out_channels=2304),
        bbox_head=dict(
            type='BBoxHeadAVA',
            dropout_ratio=0.5,
            in_channels=2304,
            num_classes=81,
            multilabel=True)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.1.csv'
ann_file_val = f'{anno_root}/ava_val_v2.1.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'

label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=10)
val_cfg = dict(interval=1)
test_cfg = dict()

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=2),
    dict(
        type='CosineAnnealingLR',
        T_max=10,
        eta_min=0,
        begin=2,
        end=10,
        by_epoch=False)
]

# optimizer
optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)

default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=40, norm_type=2)))

load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'  # noqa: E501
