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

optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.01,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.00002)
