_base_ = [
    '../slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py'  # noqa: E501
]

url = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb/'
       'slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth')
num_classes = 66
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        _delete_=True,
        type='mmaction.ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
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
    roi_head=dict(bbox_head=dict(in_channels=2304)))

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
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
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
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
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
    optimizer=dict(type='SGD', lr=0.01125, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=5, norm_type=2))
