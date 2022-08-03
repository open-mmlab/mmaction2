_base_ = ['./slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb.py']

model = dict(backbone=dict(slow_pathway=dict(lateral_norm=True)))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=34),
    dict(
        type='MultiStepLR',
        begin=34,
        end=256,
        by_epoch=True,
        milestones=[94, 154, 196],
        gamma=0.1)
]
