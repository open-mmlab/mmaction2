_base_ = [
    'slowonly_kinetics400-pretrained-r50_8xb16-8x8x1-10e_ava-kinetics-rgb.py'
]

url = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
       'slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-'
       'rgb/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_'
       'kinetics700-rgb_20221013-15b93b10.pth')

model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint=url))
