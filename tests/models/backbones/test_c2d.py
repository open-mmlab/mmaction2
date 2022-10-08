# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import C2D_R50
from ..base import generate_backbone_demo_inputs


def test_c2d_r50_backbone():
    """Test c2d_r50 backbone."""
    input_shape = (1, 3, 8, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    # c2d_r50 inference test
    c2d_r50 = C2D_R50()
    c2d_r50.init_weights()
    c2d_r50.train()
    feat = c2d_r50(imgs)
    assert feat.shape == torch.Size([1, 2048, 4, 2, 2])
