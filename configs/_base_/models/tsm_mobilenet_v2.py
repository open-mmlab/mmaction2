# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV2TSM',
        shift_div=8,
        num_segments=8,
        is_shift=True,
        pretrained='mmcls://mobilenet_v2'),
    cls_head=dict(
        type='TSMHead',
        num_segments=8,
        num_classes=400,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
