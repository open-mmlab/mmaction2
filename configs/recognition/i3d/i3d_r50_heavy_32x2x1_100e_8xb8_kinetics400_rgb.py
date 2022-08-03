_base_ = ['./i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb.py']

# model settings
model = dict(
    backbone=dict(
        inflate=(1, 1, 1, 1),
        conv1_stride_t=1,
        pool1_stride_t=1,
        with_pool2=True))
