# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import STGCN
from ..base import generate_backbone_demo_inputs


def test_stgcn_backbone():
    """Test STGCN backbone."""
    # test coco layout, spatial strategy
    input_shape = (1, 3, 300, 17, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='coco', strategy='spatial'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 17])

    # test openpose-18 layout, spatial strategy
    input_shape = (1, 3, 300, 18, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='openpose-18', strategy='spatial'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 18])

    # test ntu-rgb+d layout, spatial strategy
    input_shape = (1, 3, 300, 25, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='spatial'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 25])

    # test ntu_edge layout, spatial strategy
    input_shape = (1, 3, 300, 24, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu_edge', strategy='spatial'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 24])

    # test coco layout, uniform strategy
    input_shape = (1, 3, 300, 17, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='coco', strategy='uniform'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 17])

    # test openpose-18 layout, uniform strategy
    input_shape = (1, 3, 300, 18, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='openpose-18', strategy='uniform'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 18])

    # test ntu-rgb+d layout, uniform strategy
    input_shape = (1, 3, 300, 25, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='uniform'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 25])

    # test ntu_edge layout, uniform strategy
    input_shape = (1, 3, 300, 24, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu_edge', strategy='uniform'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 24])

    # test coco layout, distance strategy
    input_shape = (1, 3, 300, 17, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='coco', strategy='distance'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 17])

    # test openpose-18 layout, distance strategy
    input_shape = (1, 3, 300, 18, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='openpose-18', strategy='distance'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 18])

    # test ntu-rgb+d layout, distance strategy
    input_shape = (1, 3, 300, 25, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu-rgb+d', strategy='distance'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 25])

    # test ntu_edge layout, distance strategy
    input_shape = (1, 3, 300, 24, 2)
    skeletons = generate_backbone_demo_inputs(input_shape)

    stgcn = STGCN(
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='ntu_edge', strategy='distance'))
    stgcn.init_weights()
    stgcn.train()
    feat = stgcn(skeletons)
    assert feat.shape == torch.Size([2, 256, 75, 24])
