_base_ = ['slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py']

model = dict(backbone=dict(slow_pathway=dict(lateral_norm=True)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=34,
        convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        begin=0,
        end=256,
        by_epoch=True,
        milestones=[94, 154, 196],
        gamma=0.1)
]
