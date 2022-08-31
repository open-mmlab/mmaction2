_base_ = ['tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py']

# model settings
model = dict(
    backbone=dict(
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian'),
    ))
