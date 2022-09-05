_base_ = [
    'slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py'
]

model = dict(
    roi_head=dict(
        shared_head=dict(fbo_cfg=dict(type='max')),
        bbox_head=dict(in_channels=4096)))
