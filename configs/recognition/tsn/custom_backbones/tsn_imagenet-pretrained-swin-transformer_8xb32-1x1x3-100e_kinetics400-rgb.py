_base_ = ['../tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py']

model = dict(
    backbone=dict(
        type='timm.swin_base_patch4_window7_224',
        pretrained=True,
        feature_shape='NHWC',
        _delete_=True),
    cls_head=dict(in_channels=1024))
