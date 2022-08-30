# model settings
model = dict(
    type='RecognizerAudio',
    backbone=dict(type='ResNet', depth=18, in_channels=1, norm_eval=False),
    cls_head=dict(
        type='TSNAudioHead',
        num_classes=400,
        in_channels=512,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'))
