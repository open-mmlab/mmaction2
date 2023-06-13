# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import TimeSformerHead


def test_timesformer_head():
    """Test loss method, layer construction, attributes and forward function in
    timesformer head."""
    timesformer_head = TimeSformerHead(num_classes=4, in_channels=64)
    timesformer_head.init_weights()

    assert timesformer_head.num_classes == 4
    assert timesformer_head.in_channels == 64
    assert timesformer_head.init_std == 0.02

    input_shape = (2, 64)
    feat = torch.rand(input_shape)

    cls_scores = timesformer_head(feat)
    assert cls_scores.shape == torch.Size([2, 4])
