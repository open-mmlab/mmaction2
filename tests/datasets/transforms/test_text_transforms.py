# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.datasets.transforms import CLIPTokenize


class TestTextTransforms:

    @staticmethod
    def test_clip_tokenize():
        results = {'text': 'Hello, MMAction2 2.0!'}
        clip_tokenize = CLIPTokenize()
        results = clip_tokenize(results)
        assert results['text'].shape[0] == 77
        assert results['text'].dtype == torch.int32
