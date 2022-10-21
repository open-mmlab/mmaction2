model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='small', drop_path_rate=0.2),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type='MViTHead',
        in_channels=768,
        num_classes=400,
        label_smooth_eps=0.1,
        average_clips='prob'))
