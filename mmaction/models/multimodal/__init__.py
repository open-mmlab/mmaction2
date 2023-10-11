# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.utils.dependency import WITH_MULTIMODAL

if WITH_MULTIMODAL:
    from .vindlu import *  # noqa: F401,F403

else:
    from mmaction.registry import MODELS
    from mmaction.utils.dependency import register_multimodal_placeholder

    register_multimodal_placeholder(
        ['VindLUVQA', 'VindLURetrievalMC', 'VindLURetrieval'], MODELS)
