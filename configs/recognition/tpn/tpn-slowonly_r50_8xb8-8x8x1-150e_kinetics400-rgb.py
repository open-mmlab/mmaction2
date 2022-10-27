_base_ = [
    './tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb.py'
]

# model settings
model = dict(backbone=dict(pretrained=None))
