train_cfg = dict(by_epoch=True, max_epochs=150)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        milestones=[90, 130],
        gamma=0.1)
]

optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
