# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmaction.models import TPNHead


def test_tpn_head():
    """Test loss method, layer construction, attributes and forward function in
    tpn head."""
    tpn_head = TPNHead(num_classes=4, in_channels=2048)
    tpn_head.init_weights()

    assert hasattr(tpn_head, 'avg_pool2d')
    assert hasattr(tpn_head, 'avg_pool3d')
    assert isinstance(tpn_head.avg_pool3d, nn.AdaptiveAvgPool3d)
    assert tpn_head.avg_pool3d.output_size == (1, 1, 1)
    assert tpn_head.avg_pool2d is None

    input_shape = (4, 2048, 7, 7)
    feat = torch.rand(input_shape)

    # tpn head inference with num_segs
    num_segs = 2
    cls_scores = tpn_head(feat, num_segs)
    assert isinstance(tpn_head.avg_pool2d, nn.AvgPool3d)
    assert tpn_head.avg_pool2d.kernel_size == (1, 7, 7)
    assert cls_scores.shape == torch.Size([2, 4])

    # tpn head inference with no num_segs
    input_shape = (2, 2048, 3, 7, 7)
    feat = torch.rand(input_shape)
    cls_scores = tpn_head(feat)
    assert isinstance(tpn_head.avg_pool2d, nn.AvgPool3d)
    assert tpn_head.avg_pool2d.kernel_size == (1, 7, 7)
    assert cls_scores.shape == torch.Size([2, 4])
