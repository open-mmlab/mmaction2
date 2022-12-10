# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import GCNHead


def test_gcn_head():
    """Test GCNHead."""
    with pytest.raises(AssertionError):
        GCNHead(4, 5)(torch.rand((1, 2, 6, 75, 17)))

    gcn_head = GCNHead(num_classes=60, in_channels=256)
    gcn_head.init_weights()
    feat = torch.rand(1, 2, 256, 75, 25)
    cls_scores = gcn_head(feat)
    assert gcn_head.num_classes == 60
    assert gcn_head.in_channels == 256
    assert cls_scores.shape == torch.Size([1, 60])

    gcn_head = GCNHead(num_classes=60, in_channels=256, dropout=0.1)
    gcn_head.init_weights()
    feat = torch.rand(1, 2, 256, 75, 25)
    cls_scores = gcn_head(feat)
    assert gcn_head.num_classes == 60
    assert gcn_head.in_channels == 256
    assert cls_scores.shape == torch.Size([1, 60])
