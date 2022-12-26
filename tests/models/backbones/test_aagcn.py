# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import AAGCN
from mmaction.utils import register_all_modules


def test_aagcn_backbone():
    """Test AAGCN backbone."""

    register_all_modules()

    mode = 'spatial'
    batch_size, num_person, num_frames = 2, 2, 150

    # openpose-18 layout
    num_joints = 18
    model = AAGCN(graph_cfg=dict(layout='openpose', mode=mode))
    model.init_weights()
    inputs = torch.randn(batch_size, num_person, num_frames, num_joints, 3)
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 18])

    # nturgb+d layout
    num_joints = 25
    model = AAGCN(graph_cfg=dict(layout='nturgb+d', mode=mode))
    model.init_weights()
    inputs = torch.randn(batch_size, num_person, num_frames, num_joints, 3)
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 25])

    # coco layout
    num_joints = 17
    model = AAGCN(graph_cfg=dict(layout='coco', mode=mode))
    model.init_weights()
    inputs = torch.randn(batch_size, num_person, num_frames, num_joints, 3)
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 17])

    # custom settings
    # disable the attention module to degenerate AAGCN to AGCN
    model = AAGCN(
        graph_cfg=dict(layout='coco', mode=mode), gcn_attention=False)
    model.init_weights()
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 17])
