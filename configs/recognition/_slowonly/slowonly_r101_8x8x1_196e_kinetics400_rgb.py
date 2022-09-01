_base_ = ['./slowonly_r50_8x8x1_256e_kinetics400_rgb.py']

# model settings
model = dict(backbone=dict(depth=101, pretrained=None))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
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
train_cfg = dict(by_epoch=True, max_epochs=196)
# runtime settings
