_base_ = ['tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py']

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[50, 90],
        gamma=0.1)
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2))
