_base_ = ['../tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py']

checkpoint = (
    'https://download.openmmlab.com/mmclassification/v0/swin-transformer'
    '/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth')

model = dict(
    backbone=dict(
        type='mmcls.SwinTransformer',
        arch='base',
        img_size=224,
        drop_path_rate=0.5,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        _delete_=True),
    cls_head=dict(in_channels=1024))
