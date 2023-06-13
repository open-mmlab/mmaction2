_base_ = ['../tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py']

model = dict(
    backbone=dict(
        type='torchvision.densenet161', pretrained=True, _delete_=True),
    cls_head=dict(in_channels=2208))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/kinetics400/videos_train'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
