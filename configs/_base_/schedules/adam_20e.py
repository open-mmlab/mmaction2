train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[10],
        gamma=0.1)
]

optimizer = dict(
    type='Adam', lr=0.01, weight_decay=0.00001)  # this lr is used for 1 gpus

optim_wrapper = dict(
    optimizer=optimizer, clip_grad=dict(max_norm=40, norm_type=2))
