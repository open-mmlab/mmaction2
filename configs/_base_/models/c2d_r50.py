# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C2D_R50', pretrained='torchvision://resnet50', norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
