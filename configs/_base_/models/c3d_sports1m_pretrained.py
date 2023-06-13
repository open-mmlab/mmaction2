# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',  # noqa: E501
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[104, 117, 128],
        std=[1, 1, 1],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=None)
