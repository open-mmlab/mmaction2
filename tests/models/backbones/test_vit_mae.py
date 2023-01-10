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
    model.init_weights()

    assert model(x).shape == torch.Size([1, 768])
    model.eval()
    assert model(x).shape == torch.Size([1, 768])

    model = VisionTransformer(
        img_size=64,
        num_frames=8,
        use_learnable_pos_emb=True,
        drop_rate=0.1,
        use_mean_pooling=False)
    model.init_weights()

    assert model(x).shape == torch.Size([1, 768])
    model.eval()
    assert model(x).shape == torch.Size([1, 768])
