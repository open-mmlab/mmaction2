_base_ = ['../tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py']

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/' \
             'resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'

model = dict(
    backbone=dict(
        type='mmcls.ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=4,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'),
        _delete_=True))
