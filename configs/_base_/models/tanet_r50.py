# model settings
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.5],
    std=[58.395, 57.12, 57.375],
    format_shape='NCHW')

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='TANet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=8,
        tam_cfg=dict()),
    cls_head=dict(
        type='TSMHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        average_clips='prob'),
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None)
