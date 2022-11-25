# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import AGCN
from mmaction.testing import generate_backbone_demo_inputs


def test_AGCN_backbone():
    """Test AGCN backbone."""
    # test ntu-rgb+d layout, agcn strategy
    input_shape = (1, 3, 300, 25, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    agcn = AGCN(
        in_channels=3, graph_cfg=dict(layout='ntu-rgb+d', strategy='agcn'))
    agcn.init_weights()
    agcn.train()
    feat = agcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 25])
