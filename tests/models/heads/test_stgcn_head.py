# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import STGCNHead


def test_stgcn_head():
    """Test loss method, layer construction, attributes and forward function in
    stgcn head."""
    with pytest.raises(NotImplementedError):
        # spatial_type not in ['avg', 'max']
        stgcn_head = STGCNHead(
            num_classes=60, in_channels=256, spatial_type='min')
        stgcn_head.init_weights()

    # spatial_type='avg'
    stgcn_head = STGCNHead(num_classes=60, in_channels=256, spatial_type='avg')
    stgcn_head.init_weights()

    assert stgcn_head.num_classes == 60
    assert stgcn_head.in_channels == 256

    input_shape = (2, 256, 75, 17)
    feat = torch.rand(input_shape)

    cls_scores = stgcn_head(feat)
    assert cls_scores.shape == torch.Size([1, 60])

    # spatial_type='max'
    stgcn_head = STGCNHead(num_classes=60, in_channels=256, spatial_type='max')
    stgcn_head.init_weights()

    assert stgcn_head.num_classes == 60
    assert stgcn_head.in_channels == 256

    input_shape = (2, 256, 75, 17)
    feat = torch.rand(input_shape)

    cls_scores = stgcn_head(feat)
    assert cls_scores.shape == torch.Size([1, 60])
