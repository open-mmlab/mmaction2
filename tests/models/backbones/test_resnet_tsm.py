# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import pytest
import torch
import torch.nn as nn

from mmaction.models import ResNetTSM
from mmaction.models.backbones.resnet import Bottleneck
from mmaction.models.backbones.resnet_tsm import NL3DWrapper, TemporalShift
from mmaction.testing import generate_backbone_demo_inputs


class Test_ResNet_TSM(TestCase):

    def setUp(self):
        input_shape = (8, 3, 64, 64)
        self.imgs = generate_backbone_demo_inputs(input_shape)

    def test_init(self):
        with pytest.raises(NotImplementedError):
            # shift_place must be block or blockres
            resnet_tsm_50_block = ResNetTSM(50, shift_place='Block')
            resnet_tsm_50_block.init_weights()

    def test_init_from_scratch(self):
        resnet_tsm_50 = ResNetTSM(50, pretrained=None, pretrained2d=False)
        resnet_tsm_50.init_weights()

    def test_resnet_tsm_temporal_shift_blockres(self):
        # resnet_tsm with depth 50
        resnet_tsm_50 = ResNetTSM(50, pretrained='torchvision://resnet50')
        resnet_tsm_50.init_weights()
        for layer_name in resnet_tsm_50.res_layers:
            layer = getattr(resnet_tsm_50, layer_name)
            blocks = list(layer.children())
            for block in blocks:
                assert isinstance(block.conv1.conv, TemporalShift)
                assert block.conv1.conv.num_segments == resnet_tsm_50.num_segments  # noqa: E501
                assert block.conv1.conv.shift_div == resnet_tsm_50.shift_div
                assert isinstance(block.conv1.conv.net, nn.Conv2d)
        feat = resnet_tsm_50(self.imgs)
        assert feat.shape == torch.Size([8, 2048, 2, 2])

    def test_resnet_tsm_temporal_shift_block(self):
        # resnet_tsm with depth 50, no pretrained, shift_place is block
        resnet_tsm_50_block = ResNetTSM(
            50, shift_place='block', pretrained='torchvision://resnet50')
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

    def test_resnet_tsm_temporal_pool(self):
        # resnet_tsm with depth 50, no pretrained, use temporal_pool
        resnet_tsm_50_temporal_pool = ResNetTSM(
            50, temporal_pool=True, pretrained='torchvision://resnet50')
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

        feat = resnet_tsm_50_temporal_pool(self.imgs)
        assert feat.shape == torch.Size([4, 2048, 2, 2])

    def test_resnet_tsm_non_local(self):
        # resnet_tsm with non-local module
        non_local_cfg = dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')
        non_local = ((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0))
        resnet_tsm_nonlocal = ResNetTSM(
            50,
            non_local=non_local,
            non_local_cfg=non_local_cfg,
            pretrained='torchvision://resnet50')
        resnet_tsm_nonlocal.init_weights()
        for layer_name in ['layer2', 'layer3']:
            layer = getattr(resnet_tsm_nonlocal, layer_name)
            for i, _ in enumerate(layer):
                if i % 2 == 0:
                    assert isinstance(layer[i], NL3DWrapper)

        feat = resnet_tsm_nonlocal(self.imgs)
        assert feat.shape == torch.Size([8, 2048, 2, 2])

    def test_resnet_tsm_full(self):
        non_local_cfg = dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')
        non_local = ((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0))
        resnet_tsm_50_full = ResNetTSM(
            50,
            pretrained='torchvision://resnet50',
            non_local=non_local,
            non_local_cfg=non_local_cfg,
            temporal_pool=True)
        resnet_tsm_50_full.init_weights()

        input_shape = (16, 3, 32, 32)
        imgs = generate_backbone_demo_inputs(input_shape)
        feat = resnet_tsm_50_full(imgs)
        assert feat.shape == torch.Size([8, 2048, 1, 1])
