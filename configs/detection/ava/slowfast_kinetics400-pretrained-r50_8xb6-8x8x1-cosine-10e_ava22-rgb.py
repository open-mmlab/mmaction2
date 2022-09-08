_base_ = ['slowfast_kinetics400-pretrained-r50_8xb16-8x8x1-20e_ava21-rgb.py']

model = dict(
    backbone=dict(
        pretrained=(
            'https://download.openmmlab.com/mmaction/recognition/slowfast/'
            'slowfast_r50_8x8x1_256e_kinetics400_rgb/'
            'slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth')))

dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.2.csv'
ann_file_val = f'{anno_root}/ava_val_v2.2.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.2.csv'

label_file = f'{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=6,
    num_workers=8,
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
    num_workers=8,
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

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=0,
        by_epoch=True,
        begin=2,
        end=10,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001),
    clip_grad=dict(max_norm=40, norm_type=2))
