_base_ = ['tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py']

model = dict(
    backbone=dict(
        pretrained=('https://download.pytorch.org/'
                    'models/resnet101-cd907fc2.pth'),
        depth=101))
