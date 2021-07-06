# model training and testing settings
train_cfg_ = dict(
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
test_cfg_ = dict(
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
    cls_head=dict(
        type='SSNHead',
        dropout_ratio=0.,
        in_channels=2048,
        num_classes=20,
        consensus=dict(type='STPPTest', stpp_stage=(1, 1, 1)),
        use_regression=True),
    test_cfg=test_cfg_)
# dataset settings
dataset_type = 'SSNDataset'
data_root = './data/thumos14/rawframes/'
data_root_val = './data/thumos14/rawframes/'
ann_file_train = 'data/thumos14/thumos14_tag_val_proposal_list.txt'
ann_file_val = 'data/thumos14/thumos14_tag_val_proposal_list.txt'
ann_file_test = 'data/thumos14/thumos14_tag_test_proposal_list.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=True)
test_pipeline = [
    dict(
        type='SampleProposalFrames',
        clip_len=1,
        body_segments=5,
        aug_segments=(2, 2),
        aug_ratio=0.5,
        mode='test'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(340, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(
        type='Collect',
        keys=[
            'imgs', 'relative_proposal_list', 'scale_factor_list',
            'proposal_tick_list', 'reg_norm_consts'
        ],
        meta_keys=[]),
    dict(
        type='ToTensor',
        keys=[
            'imgs', 'relative_proposal_list', 'scale_factor_list',
            'proposal_tick_list', 'reg_norm_consts'
        ])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root,
        train_cfg=train_cfg_,
        test_cfg=test_cfg_,
        aug_ratio=0.5,
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=1e-6)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[200, 400])
checkpoint_config = dict(interval=5)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
# runtime settings
total_epochs = 450
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ssn_r50_1x5_450e_thumos14_rgb'
load_from = None
resume_from = None
workflow = [('train', 1)]
