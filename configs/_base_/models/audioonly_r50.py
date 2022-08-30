# model settings
model = dict(
    type='RecognizerAudio',
    backbone=dict(
        type='ResNetAudio',
        depth=50,
        pretrained=None,
        in_channels=1,
        norm_eval=False),
    cls_head=dict(
        type='TSNAudioHead',
        num_classes=400,
        in_channels=1024,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'))
