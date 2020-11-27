# model setting
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=101,
        pretrained=None,
        pretrained2d=False,
        lateral=False,
        num_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=(1, 2, 2, 1),
        temporal_strides=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    bbox_roi_extractor=dict(
        type='SingleRoIStraight3DExtractor',
        roi_layer_type='RoIAlign',
        output_size=8,
        with_temporal_pool=True),
    dropout_ratio=0.5,
    bbox_head=dict(
        type='BBoxHead', in_channels=2048, num_classes=81, multilabel=True))

train_cfg = dict(
    person_det_score_thr=0.9,
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.9,
            neg_iou_thr=0.9,
            min_pos_iou=0.9),
        sampler=dict(
            type='RandomSampler',
            num=32,
            pos_fraction=1,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=1.0,
        debug=False))
test_cfg = dict(person_det_score_thr=0.9, rcnn=dict(action_thr=0.00))

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
    dict(type='SampleAVAFrames', clip_len=8, frame_interval=8),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='EntityBoxPad'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'proposals', 'entity_boxes', 'labels'],
        meta_keys=['scores', 'entity_ids']),
    dict(
        type='ToTensor', keys=['imgs', 'proposals', 'entity_boxes', 'labels'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=8, frame_interval=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='EntityBoxPad'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'proposals'],
        meta_keys=['scores', 'img_shape']),
    dict(type='ToTensor', keys=['imgs', 'proposals'])
]

data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    # During testing, each video may have different shape
    val_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)
# this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy

lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
total_epochs = 20
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1)
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ('./work_dirs/ava/'
            'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb')
load_from = ('https://download.openmmlab.com/mmaction/recognition/slowonly/'
             'omni/'
             'slowonly_r101_omni_8x8x1_kinetics400_rgb_20200926-b5dbb701.pth')

resume_from = None
find_unused_parameters = False
