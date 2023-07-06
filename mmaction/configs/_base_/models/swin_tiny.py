# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.models import (ActionDataPreprocessor, I3DHead, Recognizer3D,
                             SwinTransformer3D)

model = dict(
    type=Recognizer3D,
    backbone=dict(
        type=SwinTransformer3D,
        arch='tiny',
        pretrained=None,
        pretrained2d=True,
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True),
    data_preprocessor=dict(
        type=ActionDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    cls_head=dict(
        type=I3DHead,
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'))
