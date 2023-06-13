_base_ = ['./i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py']

# model settings
model = dict(
    backbone=dict(
        inflate=(1, 1, 1, 1),
        conv1_stride_t=1,
        pool1_stride_t=1,
        with_pool2=True))
