# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from unittest import TestCase

import torch

from mmaction.models import MViT


class TestMViT(TestCase):

    def setUp(self):
        self.cfg = dict(arch='tiny', drop_path_rate=0.1)

    def test_structure(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            MViT(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }
            MViT(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 96,
            'num_layers': 10,
            'num_heads': 1,
            'downscale_indices': [2, 5, 8]
        }
        stage_indices = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        model = MViT(**cfg)
        self.assertEqual(model.embed_dims, 96)
        self.assertEqual(model.num_layers, 10)
        for i, block in enumerate(model.blocks):
            stage = stage_indices[i]
            self.assertEqual(block.out_dims, 96 * 2**(stage))

        # Test out_indices
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = {1: 1}
        with self.assertRaisesRegex(AssertionError, "get <class 'dict'>"):
            MViT(**cfg)
        cfg['out_scales'] = [0, 13]
        with self.assertRaisesRegex(AssertionError, 'Invalid out_scales 13'):
            MViT(**cfg)

        # Test model structure
        cfg = deepcopy(self.cfg)
        model = MViT(**cfg)
        stage_indices = [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
        self.assertEqual(len(model.blocks), 10)
        dpr_inc = 0.1 / (10 - 1)
        dpr = 0
        for i, block in enumerate(model.blocks):
            stage = stage_indices[i]
            print(i, stage)
            self.assertEqual(block.attn.num_heads, 2**stage)
            if dpr > 0:
                self.assertAlmostEqual(block.drop_path.drop_prob, dpr)
            dpr += dpr_inc

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv3d',
                mode='fan_in',
                nonlinearity='linear')
        ]
        cfg['use_abs_pos_embed'] = True
        model = MViT(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        # The pos_embed is all zero before initialize
        self.assertTrue(torch.allclose(model.pos_embed, torch.tensor(0.)))

        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
        self.assertFalse(torch.allclose(model.pos_embed, torch.tensor(0.)))

    def test_forward(self):
        imgs = torch.randn(1, 3, 6, 64, 64)

        cfg = deepcopy(self.cfg)
        model = MViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 1)
        patch_token, cls_token = outs[-1]
        self.assertEqual(patch_token.shape, (1, 768, 3, 2, 2))

        # Test forward with multi out scales
        cfg = deepcopy(self.cfg)
        cfg['out_scales'] = (0, 1, 2, 3)
        model = MViT(**cfg)
        outs = model(imgs)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for stage, out in enumerate(outs):
            stride = 2**stage
            patch_token, cls_token = out
            self.assertEqual(patch_token.shape,
                             (1, 96 * stride, 3, 16 // stride, 16 // stride))
            self.assertEqual(cls_token.shape, (1, 96 * stride))

        # Test forward with dynamic input size
        imgs1 = torch.randn(1, 3, 2, 64, 64)
        imgs2 = torch.randn(1, 3, 2, 96, 96)
        imgs3 = torch.randn(1, 3, 2, 96, 128)
        cfg = deepcopy(self.cfg)
        model = MViT(**cfg)
        for imgs in [imgs1, imgs2, imgs3]:
            outs = model(imgs)
            self.assertIsInstance(outs, tuple)
            self.assertEqual(len(outs), 1)
            patch_token, cls_token = outs[-1]
            expect_feat_shape = (math.ceil(imgs.shape[2] / 2),
                                 math.ceil(imgs.shape[3] / 32),
                                 math.ceil(imgs.shape[4] / 32))
            self.assertEqual(patch_token.shape, (1, 768, *expect_feat_shape))
            self.assertEqual(cls_token.shape, (1, 768))
