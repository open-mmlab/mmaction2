_base_ = ['slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py']

model = dict(
    backbone=dict(
        pretrained=(
            'https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly'
            '/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_'
            'kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-'
            'steplr-150e_kinetics700-rgb_20220901-f73b3e89.pth')))
