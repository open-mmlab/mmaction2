# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import STGCN


def test_stgcn_backbone():
    """Test STGCN backbone."""

    mode = 'stgcn_spatial'
    batch_size, num_person, num_frames = 2, 2, 150

    # openpose-18 layout
    num_joints = 18
    model = STGCN(graph_cfg=dict(layout='openpose', mode=mode))
    model.init_weights()
    inputs = torch.randn(batch_size, num_person, num_frames, num_joints, 3)
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 18])

    # nturgb+d layout
    num_joints = 25
    model = STGCN(graph_cfg=dict(layout='nturgb+d', mode=mode))
    model.init_weights()
    inputs = torch.randn(batch_size, num_person, num_frames, num_joints, 3)
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 25])

    # coco layout
    num_joints = 17
    model = STGCN(graph_cfg=dict(layout='coco', mode=mode))
    model.init_weights()
    inputs = torch.randn(batch_size, num_person, num_frames, num_joints, 3)
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 17])

    # custom settings
    # instantiate STGCN++
    model = STGCN(
        graph_cfg=dict(layout='coco', mode='spatial'),
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn')
    model.init_weights()
    output = model(inputs)
    assert output.shape == torch.Size([2, 2, 256, 38, 17])
