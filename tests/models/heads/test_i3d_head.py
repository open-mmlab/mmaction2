# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import I3DHead


def test_i3d_head():
    """Test loss method, layer construction, attributes and forward function in
    i3d head."""
    i3d_head = I3DHead(num_classes=4, in_channels=2048)
    i3d_head.init_weights()

    assert i3d_head.num_classes == 4
    assert i3d_head.dropout_ratio == 0.5
    assert i3d_head.in_channels == 2048
    assert i3d_head.init_std == 0.01

    assert isinstance(i3d_head.dropout, nn.Dropout)
    assert i3d_head.dropout.p == i3d_head.dropout_ratio

    assert isinstance(i3d_head.fc_cls, nn.Linear)
    assert i3d_head.fc_cls.in_features == i3d_head.in_channels
    assert i3d_head.fc_cls.out_features == i3d_head.num_classes

    assert isinstance(i3d_head.avg_pool, nn.AdaptiveAvgPool3d)
    assert i3d_head.avg_pool.output_size == (1, 1, 1)

    input_shape = (3, 2048, 4, 7, 7)
    feat = torch.rand(input_shape)

    # i3d head inference
    cls_scores = i3d_head(feat)
    assert cls_scores.shape == torch.Size([3, 4])
