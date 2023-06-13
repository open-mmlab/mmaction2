# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        fc1_bias=False,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.38, 57.38, 57.38],
        format_shape='NCTHW'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=None)
