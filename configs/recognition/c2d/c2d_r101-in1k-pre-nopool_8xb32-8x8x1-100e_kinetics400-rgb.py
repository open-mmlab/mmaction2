_base_ = ['c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.py']

model = dict(
    backbone=dict(
        pretrained=('https://download.pytorch.org/'
                    'models/resnet101-cd907fc2.pth'),
        depth=101))
