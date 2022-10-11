model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C2D_R50',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=None)
