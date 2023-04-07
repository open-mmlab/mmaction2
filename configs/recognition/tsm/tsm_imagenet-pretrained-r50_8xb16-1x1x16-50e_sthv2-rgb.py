_base_ = ['tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py']

model = dict(backbone=dict(num_segments=16), cls_head=dict(num_segments=16))

file_client_args = dict(io_backend='disk')

sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv2_flip_label_map),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        twice_sample=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

test_dataloader = dict(pipeline=test_pipeline, test_mode=True)
