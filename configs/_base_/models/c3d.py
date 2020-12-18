# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=None,
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
        init_std=0.01))

# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
