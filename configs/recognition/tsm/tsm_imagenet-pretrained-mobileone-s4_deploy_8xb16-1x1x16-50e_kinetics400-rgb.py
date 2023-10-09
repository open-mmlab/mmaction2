_base_ = [
    './tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb.py',  # noqa: E501
]

model = dict(backbone=dict(deploy=True))
