# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch
import torch.nn as nn

from mmaction.models import ResNetTSM
from mmaction.models.backbones.resnet_tsm import NL3DWrapper
from mmaction.testing import generate_backbone_demo_inputs


def test_resnet_tsm_backbone():
    """Test resnet_tsm backbone."""
    with pytest.raises(NotImplementedError):
        # shift_place must be block or blockres
        resnet_tsm_50_block = ResNetTSM(50, shift_place='Block')
        resnet_tsm_50_block.init_weights()

    from mmaction.models.backbones.resnet import Bottleneck
    from mmaction.models.backbones.resnet_tsm import TemporalShift

    input_shape = (8, 3, 64, 64)
    imgs = generate_backbone_demo_inputs(input_shape)

    # resnet_tsm with depth 50
    resnet_tsm_50 = ResNetTSM(50)
    resnet_tsm_50.init_weights()
    for layer_name in resnet_tsm_50.res_layers:
        layer = getattr(resnet_tsm_50, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block.conv1.conv, TemporalShift)
            assert block.conv1.conv.num_segments == resnet_tsm_50.num_segments
            assert block.conv1.conv.shift_div == resnet_tsm_50.shift_div
            assert isinstance(block.conv1.conv.net, nn.Conv2d)

    # resnet_tsm with depth 50, no pretrained, shift_place is block
    resnet_tsm_50_block = ResNetTSM(50, shift_place='block')
    resnet_tsm_50_block.init_weights()
    for layer_name in resnet_tsm_50_block.res_layers:
        layer = getattr(resnet_tsm_50_block, layer_name)
        blocks = list(layer.children())
        for block in blocks:
            assert isinstance(block, TemporalShift)
            assert block.num_segments == resnet_tsm_50_block.num_segments
            assert block.num_segments == resnet_tsm_50_block.num_segments
            assert block.shift_div == resnet_tsm_50_block.shift_div
            assert isinstance(block.net, Bottleneck)

    # resnet_tsm with depth 50, no pretrained, use temporal_pool
    resnet_tsm_50_temporal_pool = ResNetTSM(50, temporal_pool=True)
    resnet_tsm_50_temporal_pool.init_weights()
    for layer_name in resnet_tsm_50_temporal_pool.res_layers:
        layer = getattr(resnet_tsm_50_temporal_pool, layer_name)
        blocks = list(layer.children())

        if layer_name == 'layer2':
            assert len(blocks) == 2
            assert isinstance(blocks[1], nn.MaxPool3d)
            blocks = copy.deepcopy(blocks[0])

        for block in blocks:
            assert isinstance(block.conv1.conv, TemporalShift)
            if layer_name == 'layer1':
                assert block.conv1.conv.num_segments == \
                       resnet_tsm_50_temporal_pool.num_segments
            else:
                assert block.conv1.conv.num_segments == \
                       resnet_tsm_50_temporal_pool.num_segments // 2
            assert block.conv1.conv.shift_div == resnet_tsm_50_temporal_pool.shift_div  # noqa: E501
            assert isinstance(block.conv1.conv.net, nn.Conv2d)

    # resnet_tsm with non-local module
    non_local_cfg = dict(
        sub_sample=True,
        use_scale=False,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        mode='embedded_gaussian')
    non_local = ((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0))
    resnet_tsm_nonlocal = ResNetTSM(
        50, non_local=non_local, non_local_cfg=non_local_cfg)
    resnet_tsm_nonlocal.init_weights()
    for layer_name in ['layer2', 'layer3']:
        layer = getattr(resnet_tsm_nonlocal, layer_name)
        for i, _ in enumerate(layer):
            if i % 2 == 0:
                assert isinstance(layer[i], NL3DWrapper)

    resnet_tsm_50_full = ResNetTSM(
        50,
        non_local=non_local,
        non_local_cfg=non_local_cfg,
        temporal_pool=True)
    resnet_tsm_50_full.init_weights()

    # TSM forword
    feat = resnet_tsm_50(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    # TSM with non-local forward
    feat = resnet_tsm_nonlocal(imgs)
    assert feat.shape == torch.Size([8, 2048, 2, 2])

    # TSM with temporal pool forward
    feat = resnet_tsm_50_temporal_pool(imgs)
    assert feat.shape == torch.Size([4, 2048, 2, 2])

    # TSM with temporal pool + non-local forward
    input_shape = (16, 3, 32, 32)
    imgs = generate_backbone_demo_inputs(input_shape)
    feat = resnet_tsm_50_full(imgs)
    assert feat.shape == torch.Size([8, 2048, 1, 1])
