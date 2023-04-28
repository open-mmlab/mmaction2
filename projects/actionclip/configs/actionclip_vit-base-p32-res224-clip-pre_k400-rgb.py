_base_ = 'mmaction::_base_/default_runtime.py'

custom_imports = dict(imports='models')

num_segs = 8

model = dict(
    type='ActionClip',
    clip_arch='ViT-B/32',
    num_adapter_segs=num_segs,
    num_adapter_layers=6,
    labels_or_label_file='configs/label_map_k400.txt',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.771, 116.746, 104.093],
        std=[68.500, 66.632, 70.323],
        format_shape='NCHW'))

dataset_type = 'VideoDataset'
data_root_val = 'data/kinetics400/videos_val'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(io_backend='disk')
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=num_segs,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='AccMetric')
