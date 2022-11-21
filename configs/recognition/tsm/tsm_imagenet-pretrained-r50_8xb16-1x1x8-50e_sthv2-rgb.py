_base_ = ['tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py']

# model settings
model = dict(cls_head=dict(num_classes=174, dropout_ratio=0.5))

# dataset settings
data_root = 'data/sthv2/videos'
ann_file_train = 'data/sthv2/sthv2_train_list_videos.txt'
ann_file_val = 'data/sthv2/sthv2_val_list_videos.txt'


test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
        twice_sample=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root)))

val_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root)))

test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        data_prefix=dict(video=data_root)))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[25, 45],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.02, weight_decay=0.0005))
