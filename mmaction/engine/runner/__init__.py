# Copyright (c) OpenMMLab. All rights reserved.
from .multi_loop import MultiLoaderEpochBasedTrainLoop
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop

__all__ = [
    'MultiLoaderEpochBasedTrainLoop', 'RetrievalValLoop', 'RetrievalTestLoop'
]
