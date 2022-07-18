# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import X3DHead


def test_x3d_head():
    """Test loss method, layer construction, attributes and forward function in
    x3d head."""
    x3d_head = X3DHead(in_channels=432, num_classes=4, fc1_bias=False)
    x3d_head.init_weights()

    assert x3d_head.num_classes == 4
    assert x3d_head.dropout_ratio == 0.5
    assert x3d_head.in_channels == 432
    assert x3d_head.init_std == 0.01

    assert isinstance(x3d_head.dropout, nn.Dropout)
    assert x3d_head.dropout.p == x3d_head.dropout_ratio

    assert isinstance(x3d_head.fc1, nn.Linear)
    assert x3d_head.fc1.in_features == x3d_head.in_channels
    assert x3d_head.fc1.out_features == x3d_head.mid_channels
    assert x3d_head.fc1.bias is None

    assert isinstance(x3d_head.fc2, nn.Linear)
    assert x3d_head.fc2.in_features == x3d_head.mid_channels
    assert x3d_head.fc2.out_features == x3d_head.num_classes

    assert isinstance(x3d_head.pool, nn.AdaptiveAvgPool3d)
    assert x3d_head.pool.output_size == (1, 1, 1)

    input_shape = (3, 432, 4, 7, 7)
    feat = torch.rand(input_shape)

    # i3d head inference
    cls_scores = x3d_head(feat)
    assert cls_scores.shape == torch.Size([3, 4])
