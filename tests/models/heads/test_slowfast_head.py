# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import SlowFastHead


def test_slowfast_head():
    """Test loss method, layer construction, attributes and forward function in
    slowfast head."""
    sf_head = SlowFastHead(num_classes=4, in_channels=2304)
    sf_head.init_weights()

    assert sf_head.num_classes == 4
    assert sf_head.dropout_ratio == 0.8
    assert sf_head.in_channels == 2304
    assert sf_head.init_std == 0.01

    assert isinstance(sf_head.dropout, nn.Dropout)
    assert sf_head.dropout.p == sf_head.dropout_ratio

    assert isinstance(sf_head.fc_cls, nn.Linear)
    assert sf_head.fc_cls.in_features == sf_head.in_channels
    assert sf_head.fc_cls.out_features == sf_head.num_classes

    assert isinstance(sf_head.avg_pool, nn.AdaptiveAvgPool3d)
    assert sf_head.avg_pool.output_size == (1, 1, 1)

    input_shape = (3, 2048, 32, 7, 7)
    feat_slow = torch.rand(input_shape)

    input_shape = (3, 256, 4, 7, 7)
    feat_fast = torch.rand(input_shape)

    sf_head = SlowFastHead(num_classes=4, in_channels=2304)
    cls_scores = sf_head((feat_slow, feat_fast))
    assert cls_scores.shape == torch.Size([3, 4])
