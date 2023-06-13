# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import TSMHead


def test_tsm_head():
    """Test loss method, layer construction, attributes and forward function in
    tsm head."""
    tsm_head = TSMHead(num_classes=4, in_channels=2048)
    tsm_head.init_weights()

    assert tsm_head.num_classes == 4
    assert tsm_head.dropout_ratio == 0.8
    assert tsm_head.in_channels == 2048
    assert tsm_head.init_std == 0.001
    assert tsm_head.consensus.dim == 1
    assert tsm_head.spatial_type == 'avg'

    assert isinstance(tsm_head.dropout, nn.Dropout)
    assert tsm_head.dropout.p == tsm_head.dropout_ratio

    assert isinstance(tsm_head.fc_cls, nn.Linear)
    assert tsm_head.fc_cls.in_features == tsm_head.in_channels
    assert tsm_head.fc_cls.out_features == tsm_head.num_classes

    assert isinstance(tsm_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert tsm_head.avg_pool.output_size == 1

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tsm head inference with no init
    num_segs = input_shape[0]
    cls_scores = tsm_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # tsm head inference with init
    tsm_head = TSMHead(num_classes=4, in_channels=2048, temporal_pool=True)
    tsm_head.init_weights()
    cls_scores = tsm_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([2, 4])
