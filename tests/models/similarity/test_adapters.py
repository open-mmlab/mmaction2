# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import SimpleMeanAdapter, TransformerAdapter


def test_transformer_adapter():
    """Test transformer adapter."""
    with pytest.raises(RuntimeError):
        num_segs_model = 8
        num_segs_features = 9
        adapter = TransformerAdapter(
            num_segs=num_segs_model,
            transformer_width=64,
            transformer_heads=8,
            transformer_layers=2)
        features = torch.randn(2, num_segs_features, 64)
        adapter(features)

    num_segs = 8
    adapter = TransformerAdapter(
        num_segs=num_segs,
        transformer_width=64,
        transformer_heads=8,
        transformer_layers=2)
    adapter.init_weights()
    features = torch.randn(2, num_segs, 64)
    adapted_features = adapter(features)
    assert adapted_features.shape == torch.Size([2, 64])


def test_simple_mean_adapter():
    """Test simple mean adapter."""

    adapter = SimpleMeanAdapter(dim=1)
    features = torch.randn(2, 8, 64)
    adapted_features = adapter(features)
    assert adapted_features.shape == torch.Size([2, 64])

    adapter = SimpleMeanAdapter(dim=(1, 2))
    features = torch.randn(2, 8, 2, 64)
    adapted_features = adapter(features)
    assert adapted_features.shape == torch.Size([2, 64])
