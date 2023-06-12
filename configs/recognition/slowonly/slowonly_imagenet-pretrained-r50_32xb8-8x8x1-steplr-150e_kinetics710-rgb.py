_base_ = [('slowonly_imagenet-pretrained-r50_16xb16-'
           '4x16x1-steplr-150e_kinetics700-rgb.py')]

model = dict(cls_head=dict(num_classes=710))

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

k400_data_root = 'data/kinetics400/videos_train'
k600_data_root = 'data/kinetics600/videos'
k700_data_root = 'data/kinetics700/videos'
k400_data_root_val = 'data/kinetics400/videos_val'
k600_data_root_val = k600_data_root
k700_data_root_val = k700_data_root

k400_ann_file_train = 'data/kinetics710/k400_train_list_videos.txt'
k600_ann_file_train = 'data/kinetics710/k600_train_list_videos.txt'
k700_ann_file_train = 'data/kinetics710/k700_train_list_videos.txt'

k400_ann_file_val = 'data/kinetics710/k400_val_list_videos.txt'
k600_ann_file_val = 'data/kinetics710/k600_val_list_videos.txt'
k700_ann_file_val = 'data/kinetics710/k700_val_list_videos.txt'

k400_trainset = dict(
    type='VideoDataset',
    ann_file=k400_ann_file_train,
    data_prefix=dict(video=k400_data_root),
    pipeline=train_pipeline)
k600_trainset = dict(
    type='VideoDataset',
    ann_file=k600_ann_file_train,
    data_prefix=dict(video=k600_data_root),
    pipeline=train_pipeline)
k700_trainset = dict(
    type='VideoDataset',
    ann_file=k700_ann_file_train,
    data_prefix=dict(video=k700_data_root),
    pipeline=train_pipeline)

k400_valset = dict(
    type='VideoDataset',
    ann_file=k400_ann_file_val,
    data_prefix=dict(video=k400_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)
k600_valset = dict(
    type='VideoDataset',
    ann_file=k600_ann_file_val,
    data_prefix=dict(video=k600_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)
k700_valset = dict(
    type='VideoDataset',
    ann_file=k700_ann_file_val,
    data_prefix=dict(video=k700_data_root_val),
    pipeline=val_pipeline,
    test_mode=True)

k400_testset = k400_valset.copy()
k600_testset = k600_valset.copy()
k700_testset = k700_valset.copy()
k400_testset['pipeline'] = test_pipeline
k600_testset['pipeline'] = test_pipeline
k700_testset['pipeline'] = test_pipeline

k710_trainset = dict(
    type='ConcatDataset',
    datasets=[k400_trainset, k600_trainset, k700_trainset],
    _delete_=True)
k710_valset = dict(
    type='ConcatDataset',
    datasets=[k400_valset, k600_valset, k700_valset],
    _delete_=True)
k710_testset = dict(
    type='ConcatDataset',
    datasets=[k400_testset, k600_testset, k700_testset],
    _delete_=True,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=k710_trainset)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=k710_valset)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=k710_testset)
