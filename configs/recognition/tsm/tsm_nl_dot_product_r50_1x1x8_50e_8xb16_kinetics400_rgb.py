_base_ = ['tsm_r50_1x1x8_50e_8xb16_kinetics400_rgb.py']

# model settings
model = dict(
    backbone=dict(
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='dot_product'),
    ))
