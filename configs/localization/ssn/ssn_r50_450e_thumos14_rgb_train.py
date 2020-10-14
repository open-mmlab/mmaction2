# model training and testing settings
train_cfg = dict(
    ssn=dict(
        assigner=dict(
            positive_iou_threshold=0.7,
            background_iou_threshold=0.01,
            incomplete_iou_threshold=0.3,
            background_coverage_threshold=0.02,
            incomplete_overlap_threshold=0.01),
        sampler=dict(
            num_per_video=8,
            positive_ratio=1,
            background_ratio=1,
            incomplete_ratio=6,
            add_gt_as_proposals=True),
        loss_weight=dict(comp_loss_weight=0.1, reg_loss_weight=0.1),
        debug=False))
test_cfg = dict(
    ssn=dict(
        sampler=dict(test_interval=6, batch_size=16),
        evaluater=dict(
            top_k=2000,
            nms=0.2,
            softmax_before_filter=True,
            cls_score_dict=None,
            cls_top_k=2)))
# model settings
model = dict(
    type='SSN',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        partial_bn=True),
    spatial_type='avg',
    dropout_ratio=0.8,
    loss_cls=dict(type='SSNLoss'),
    cls_head=dict(
        type='SSNHead',
        dropout_ratio=0.,
        in_channels=2048,
        num_classes=20,
        consensus=dict(
            type='STPPTrain',
            stpp_stage=(1, 1, 1),
            num_segments_list=(2, 5, 2)),
        use_regression=True),
    train_cfg=train_cfg)
# dataset settings
dataset_type = 'SSNDataset'
data_root = './data/thumos14/rawframes/'
data_root_val = './data/thumos14/rawframes/'
ann_file_train = 'data/thumos14/thumos14_tag_val_proposal_list.txt'
ann_file_val = 'data/thumos14/thumos14_tag_val_proposal_list.txt'
ann_file_test = 'data/thumos14/thumos14_tag_test_proposal_list.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=True)
train_pipeline = [
    dict(
        type='SampleProposalFrames',
        clip_len=1,
        body_segments=5,
        aug_segments=(2, 2),
        aug_ratio=0.5),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(340, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(
        type='Collect',
        keys=[
            'imgs', 'reg_targets', 'proposal_scale_factor', 'proposal_labels',
            'proposal_type'
        ],
        meta_keys=[]),
    dict(
        type='ToTensor',
        keys=[
            'imgs', 'reg_targets', 'proposal_scale_factor', 'proposal_labels',
            'proposal_type'
        ])
]
val_pipeline = [
    dict(
        type='SampleProposalFrames',
        clip_len=1,
        body_segments=5,
        aug_segments=(2, 2),
        aug_ratio=0.5),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(340, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NPTCHW'),
    dict(
        type='Collect',
        keys=[
            'imgs', 'reg_targets', 'proposal_scale_factor', 'proposal_labels',
            'proposal_type'
        ],
        meta_keys=[]),
    dict(
        type='ToTensor',
        keys=[
            'imgs', 'reg_targets', 'proposal_scale_factor', 'proposal_labels',
            'proposal_type'
        ])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        body_segments=5,
        aug_segments=(2, 2),
        aug_ratio=0.5,
        test_mode=False,
        verbose=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        body_segments=5,
        aug_segments=(2, 2),
        aug_ratio=0.5,
        test_mode=False,
        pipeline=val_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=1e-6)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[200, 400])
checkpoint_config = dict(interval=5)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
# runtime settings
total_epochs = 450
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssn_r50_1x5_450e_thumos14_rgb'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
