# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmaction.models import FeatureHead
from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_recognizer_cfg
from mmaction.utils import register_all_modules


class TestFeatureHead(TestCase):

    def test_2d_recognizer(self):
        register_all_modules()
        config = get_recognizer_cfg(
            'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'  # noqa: E501
        )
        config.model['backbone']['pretrained'] = None
        config.model['cls_head'] = dict(
            type='FeatureHead', average_clips='score')

        recognizer = MODELS.build(config.model)

        input_shape = [3, 3, 32, 32]
        data_batch = {
            'inputs': [torch.randint(0, 256, input_shape)],
            'data_samples': [ActionDataSample().set_gt_label(2)]
        }
        feat = recognizer.test_step(data_batch)
        assert isinstance(feat, torch.Tensor)
        assert feat.shape == torch.Size([1, 2048])

    def test_3d_recognizer(self):
        register_all_modules()
        config = get_recognizer_cfg(
            'slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py')
        config.model['backbone']['pretrained'] = None
        config.model['backbone']['pretrained2d'] = False
        config.model['cls_head'] = dict(
            type='FeatureHead', average_clips='score')

        recognizer = MODELS.build(config.model)
        input_shape = [1, 3, 4, 32, 32]
        data_batch = {
            'inputs': [torch.randint(0, 256, input_shape)],
            'data_samples': [ActionDataSample().set_gt_label(2)]
        }
        feat = recognizer.test_step(data_batch)
        assert isinstance(feat, torch.Tensor)
        assert feat.shape == torch.Size([1, 2048])

    def test_3d_backbone(self):
        with pytest.raises(NotImplementedError):
            head = FeatureHead(spatial_type='test')

        head = FeatureHead(average_clips='score')
        x = torch.rand(1, 64, 2, 7, 7)
        feat = head(x)
        assert feat.shape == torch.Size([1, 64])

        head = FeatureHead(spatial_type=None, average_clips='score')
        feat = head(x)
        assert feat.shape == torch.Size([1, 64, 7, 7])

        head = FeatureHead(temporal_type=None, average_clips='score')
        feat = head(x)
        assert feat.shape == torch.Size([1, 64, 2])

        head = FeatureHead(
            spatial_type=None, temporal_type=None, average_clips='score')
        feat = head(x)
        assert feat.shape == torch.Size([1, 64, 2, 7, 7])

    def test_slowfast_backbone(self):
        head = FeatureHead(backbone_name='slowfast', average_clips='score')
        x_slow = torch.rand(1, 64, 2, 7, 7)
        x_fast = torch.rand(1, 32, 6, 7, 7)
        x = (x_slow, x_fast)
        feat = head(x)
        assert feat.shape == torch.Size([1, 96])

        head = FeatureHead(
            backbone_name='slowfast', spatial_type=None, average_clips='score')
        feat = head(x)
        assert feat.shape == torch.Size([1, 96, 7, 7])

        with pytest.raises(AssertionError):
            head = FeatureHead(
                backbone_name='slowfast',
                temporal_type=None,
                average_clips='score')
            feat = head(x)

    def test_2d_backbone(self):
        head = FeatureHead(average_clips='score')
        x = torch.rand(2, 64, 7, 7)
        with pytest.raises(AssertionError):
            feat = head(x)

        feat = head(x, num_segs=2)
        assert feat.shape == torch.Size([1, 64])

        x = torch.rand(2, 64, 7, 7)
        head = FeatureHead(spatial_type=None, average_clips='score')
        feat = head(x, num_segs=2)
        assert feat.shape == torch.Size([1, 64, 7, 7])

        head = FeatureHead(temporal_type=None, average_clips='score')
        feat = head(x, num_segs=2)
        assert feat.shape == torch.Size([1, 2, 64])

    def test_tsm_backbone(self):
        head = FeatureHead(backbone_name='tsm', average_clips='score')
        x = torch.rand(2, 64, 7, 7)
        with pytest.raises(AssertionError):
            feat = head(x)
        with pytest.raises(AssertionError):
            feat = head(x, num_segs=2)

        head = FeatureHead(num_segments=2, average_clips='score')
        feat = head(x, num_segs=2)
        assert feat.shape == torch.Size([1, 64])

        x = torch.rand(2, 64, 7, 7)
        head = FeatureHead(
            num_segments=2, spatial_type=None, average_clips='score')
        feat = head(x, num_segs=2)
        assert feat.shape == torch.Size([1, 64, 7, 7])

    def test_gcn_backbone(self):
        # N, M, C, T, V
        head = FeatureHead(backbone_name='gcn', average_clips='score')
        x = torch.rand(1, 5, 64, 2, 7)
        feat = head(x)
        assert feat.shape == torch.Size([1, 64])
