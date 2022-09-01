_base_ = ['slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb.py']

# model settings
model = dict(backbone=dict(depth=101, pretrained=None))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=196, val_begin=1, val_interval=5)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=34),
    dict(
        type='CosineAnnealingLR',
        T_max=162,
        eta_min=0,
        by_epoch=True,
        begin=34,
        end=196)
]
