_base_ = ['../../_base_/models/x3d.py', '../../_base_/default_runtime.py']

# dataset settings
dataset_type = 'VideoDataset'
data_root_val = 'data/kinetics400/videos_val'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        target_fps=30,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='AccMetric')
test_cfg = dict(type='TestLoop')
