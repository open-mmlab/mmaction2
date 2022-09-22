# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import VisionTransformer


def test_vit_backbone():
    """Test vit backbone."""
    x = torch.randn(1, 3, 8, 64, 64)
    model = VisionTransformer(
        img_size=64,
        num_frames=8,
        qkv_bias=True,
        drop_path_rate=0.2,
        init_values=0.1)

    assert model(x).shape == torch.Size([1, 768])
    model.eval()
    assert model(x).shape == torch.Size([1, 768])

    model = VisionTransformer(
        img_size=64,
        num_frames=8,
        use_learnable_pos_emb=True,
        use_mean_pooling=False)

    assert model(x).shape == torch.Size([1, 768])
    model.eval()
    assert model(x).shape == torch.Size([1, 768])
