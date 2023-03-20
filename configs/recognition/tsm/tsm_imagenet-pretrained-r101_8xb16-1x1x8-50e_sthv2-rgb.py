_base_ = ['tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py']

# model settings
r101_checkpoint = 'torchvision://resnet101'

model = dict(backbone=dict(pretrained=r101_checkpoint, depth=101))
