# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn

from mmaction.models import MViTHead


class TestMViTHead(TestCase):
    DEFAULT_ARGS = dict(in_channels=768, num_classes=5)
    fake_feats = ([torch.rand(4, 768, 3, 2, 2), torch.rand(4, 768)], )

    def test_init(self):
        head = MViTHead(**self.DEFAULT_ARGS)
        head.init_weights()
        self.assertEqual(head.dropout.p, head.dropout_ratio)
        self.assertIsInstance(head.fc_cls, nn.Linear)
        self.assertEqual(head.num_classes, 5)
        self.assertEqual(head.dropout_ratio, 0.5)
        self.assertEqual(head.in_channels, 768)
        self.assertEqual(head.init_std, 0.02)

    def test_pre_logits(self):
        head = MViTHead(**self.DEFAULT_ARGS)
        pre_logits = head.pre_logits(self.fake_feats)
        self.assertIs(pre_logits, self.fake_feats[-1][1])

    def test_forward(self):
        head = MViTHead(**self.DEFAULT_ARGS)
        cls_score = head(self.fake_feats)
        self.assertEqual(cls_score.shape, (4, 5))
