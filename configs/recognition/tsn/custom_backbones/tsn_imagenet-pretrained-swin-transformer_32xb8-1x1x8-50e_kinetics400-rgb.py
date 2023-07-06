_base_ = ['../tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py']

model = dict(
    backbone=dict(
        type='timm.swin_base_patch4_window7_224',
        pretrained=True,
        _delete_=True),
    cls_head=dict(in_channels=1024))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]
