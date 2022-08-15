_base_ = ['../tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb.py']

model = dict(
    backbone=dict(
        type='timm.swin_base_patch4_window7_224',
        pretrained=True,
        _delete_=True),
    cls_head=dict(in_channels=1024))
