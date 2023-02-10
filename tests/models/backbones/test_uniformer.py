# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import UniFormer
from mmaction.testing import generate_backbone_demo_inputs


def test_uniformer_backbone():
    """Test uniformer backbone."""
    input_shape = (1, 3, 16, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        drop_path_rate=0.1)
    model.init_weights()

    model.eval()
    assert model(imgs).shape == torch.Size([1, 512, 8, 2, 2])
