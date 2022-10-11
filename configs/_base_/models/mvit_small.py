model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT', arch='small', drop_path_rate=0.2),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        blending=dict(
            type='RandomBatchAugment',
            augments=[
                dict(type='MixupBlending', alpha=0.8, num_classes=400),
                dict(type='CutmixBlending', alpha=1, num_classes=400)
            ]),
        format_shape='NCTHW'),
    cls_head=dict(
        type='MVitHead',
        in_channels=768,
        num_classes=400,
        label_smooth_eps=0.1,
        average_clips='prob'))
