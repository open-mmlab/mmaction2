_base_ = [
    '../tsm_imagenet-pretrained-mobilenetone-s4_8xb16-1x1x16-50e_kinetics400-rgb.py',
]

model = dict(backbone=dict(deploy=True))