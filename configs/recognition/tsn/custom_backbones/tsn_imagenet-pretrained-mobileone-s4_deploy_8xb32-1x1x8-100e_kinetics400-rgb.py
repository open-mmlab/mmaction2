_base_ = [
    './tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb.py'  # noqa: E501
]

model = dict(backbone=dict(deploy=True))
