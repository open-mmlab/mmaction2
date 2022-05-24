_base_ = ['./tsm_r50_1x1x8_50e_sthv1_rgb.py']

# model settings
model = dict(backbone=dict(pretrained='torchvision://resnet101', depth=101))