# model settings
model = dict(
    type='AudioRecognizer',
    backbone=dict(
        type='ResNetAudio',
        depth=50,
        pretrained=None,
        in_channels=1,
        norm_eval=False),
    cls_head=dict(
        type='AudioTSNHead',
        num_classes=400,
        in_channels=1024,
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
