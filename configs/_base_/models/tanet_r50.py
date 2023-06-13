# model settings
model = dict(
    type='Recognizer2D',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.5],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    backbone=dict(
        type='TANet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=8,
        tam_cfg=None),
    cls_head=dict(
        type='TSMHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        average_clips='prob'))
