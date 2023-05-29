# Copyright (c) OpenMMLab. All rights reserved.
from .adapters import SimpleMeanAdapter, TransformerAdapter
from .clip_similarity import CLIPSimilarity

__all__ = ['CLIPSimilarity', 'TransformerAdapter', 'SimpleMeanAdapter']
