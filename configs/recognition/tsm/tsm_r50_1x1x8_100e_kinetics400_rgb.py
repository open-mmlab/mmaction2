_base_ = ['./tsm_r50_1x1x8_50e_kinetics400_rgb.py']

train_cfg = dict(by_epoch=True, max_epochs=100)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[40, 80],
        gamma=0.1)
]