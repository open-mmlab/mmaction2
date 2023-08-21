# Copyright (c) OpenMMLab. All rights reserved.
from .multi_loop import MultiLoaderEpochBasedTrainLoop
from .retrieval_loop import RetrievalValLoop, RetrievalTestLoop

__all__ = ['MultiLoaderEpochBasedTrainLoop', 'RetrievalValLoop', 'RetrievalTestLoop']
