_base_ = ['./i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py']

# model settings
model = dict(
    backbone=dict(
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='gaussian')))

# runtime settings
work_dir = './work_dirs/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/'
