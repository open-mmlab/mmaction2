# Copyright (c) OpenMMLab. All rights reserved.
from .clip_similarity import CLIPSimilarity
from .adapters import TransformerAdapter, SimpleMeanAdapter

__all__ = ['CLIPSimilarity', 'TransformerAdapter', 'SimpleMeanAdapter']
