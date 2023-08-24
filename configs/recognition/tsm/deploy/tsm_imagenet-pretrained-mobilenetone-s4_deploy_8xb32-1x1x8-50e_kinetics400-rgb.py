_base_ = [
    '../tsm_imagenet-pretrained-mobilenetone-s4_8xb32-1x1x8-50e_kinetics400-rgb.py',
]

model = dict(backbone=dict(deploy=True))