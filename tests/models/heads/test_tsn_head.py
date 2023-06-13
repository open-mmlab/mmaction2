# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import TSNHead


def test_tsn_head():
    """Test loss method, layer construction, attributes and forward function in
    tsn head."""
    tsn_head = TSNHead(num_classes=4, in_channels=2048)
    tsn_head.init_weights()

    assert tsn_head.num_classes == 4
    assert tsn_head.dropout_ratio == 0.4
    assert tsn_head.in_channels == 2048
    assert tsn_head.init_std == 0.01
    assert tsn_head.consensus.dim == 1
    assert tsn_head.spatial_type == 'avg'

    assert isinstance(tsn_head.dropout, nn.Dropout)
    assert tsn_head.dropout.p == tsn_head.dropout_ratio

    assert isinstance(tsn_head.fc_cls, nn.Linear)
    assert tsn_head.fc_cls.in_features == tsn_head.in_channels
    assert tsn_head.fc_cls.out_features == tsn_head.num_classes

    assert isinstance(tsn_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert tsn_head.avg_pool.output_size == (1, 1)

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tsn head inference
    num_segs = input_shape[0]
    cls_scores = tsn_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])

    # Test multi-class recognition
    multi_tsn_head = TSNHead(
        num_classes=4,
        in_channels=2048,
        loss_cls=dict(type='BCELossWithLogits', loss_weight=160.0),
        multi_class=True,
        label_smooth_eps=0.01)
    multi_tsn_head.init_weights()
    assert multi_tsn_head.num_classes == 4
    assert multi_tsn_head.dropout_ratio == 0.4
    assert multi_tsn_head.in_channels == 2048
    assert multi_tsn_head.init_std == 0.01
    assert multi_tsn_head.consensus.dim == 1

    assert isinstance(multi_tsn_head.dropout, nn.Dropout)
    assert multi_tsn_head.dropout.p == multi_tsn_head.dropout_ratio

    assert isinstance(multi_tsn_head.fc_cls, nn.Linear)
    assert multi_tsn_head.fc_cls.in_features == multi_tsn_head.in_channels
    assert multi_tsn_head.fc_cls.out_features == multi_tsn_head.num_classes

    assert isinstance(multi_tsn_head.avg_pool, nn.AdaptiveAvgPool2d)
    assert multi_tsn_head.avg_pool.output_size == (1, 1)

    input_shape = (8, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # multi-class tsn head inference
    num_segs = input_shape[0]
    cls_scores = tsn_head(feat, num_segs)
    assert cls_scores.shape == torch.Size([1, 4])
