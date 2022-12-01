_base_ = [
    'slowonly_kinetics700-pretrained-r50_8xb16-8x8x1-10e_ava-kinetics-rgb.py'
]

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(with_global=True, temporal_pool_mode='max'),
        bbox_head=dict(in_channels=4096)))
