_base_ = [
    'c2d_nopool_imagenet-pretrained-r50_8xb32-16x4x1-100e_kinetics400-rgb.py'
]

model = dict(
    backbone=dict(
        pretrained=('https://download.pytorch.org/'
                    'models/resnet101-cd907fc2.pth'),
        depth=101))
