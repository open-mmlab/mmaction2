_base_ = [
    'i3d_imagenet-pretrained-r50-nl-dot-product_' +
    '8xb8-32x2x1-100e_kinetics400-rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='gaussian')))
