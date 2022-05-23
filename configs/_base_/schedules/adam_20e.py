train_cfg = dict(by_epoch=True, max_epochs=20)

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