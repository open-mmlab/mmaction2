_base_ = ['tsm_r50_dense_1x1x8_100e_kinetics400_rgb.py']

train_cfg = dict(by_epoch=True, max_epochs=50)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]