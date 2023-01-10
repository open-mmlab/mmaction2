# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import C3D
from mmaction.testing import generate_backbone_demo_inputs


def test_c3d_backbone():
    """Test c3d backbone."""
    input_shape = (1, 3, 16, 24, 24)
    imgs = generate_backbone_demo_inputs(input_shape)

    # c3d inference test
    c3d = C3D(out_dim=512)
    c3d.init_weights()
    c3d.train()
    feat = c3d(imgs)
    assert feat.shape == torch.Size([1, 4096])

    # c3d with bn inference test
    c3d_bn = C3D(out_dim=512, norm_cfg=dict(type='BN3d'))
    c3d_bn.init_weights()
    c3d_bn.train()
    feat = c3d_bn(imgs)
    assert feat.shape == torch.Size([1, 4096])
