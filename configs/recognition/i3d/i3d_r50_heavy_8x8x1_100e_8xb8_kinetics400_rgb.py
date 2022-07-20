_base_ = ['i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb.py']

# model settings
model = dict(
    backbone=dict(
        inflate=(1, 1, 1, 1),
        conv1_stride_t=1,
        pool1_stride_t=1,
        with_pool2=True))

# dataset settings
dataset_type = 'VideoDataset'
root = './data/kinetics400/'
data_root = root + 'videos_train'
data_root_val = root + 'videos_val'
data_root_test = data_root_val

ann_file_train = root + 'kinetics400_train_list_videos.txt'
ann_file_val = root + 'kinetics400_val_list_videos.txt'
ann_file_test = ann_file_val

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
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
    dict(type='DecordInit'),
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
