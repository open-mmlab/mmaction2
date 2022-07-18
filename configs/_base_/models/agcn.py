model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AGCN',
        in_channels=3,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='agcn')),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)
