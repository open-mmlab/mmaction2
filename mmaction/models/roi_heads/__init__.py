# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmdet.registry import MODELS as MMDET_MODELS

    from .bbox_heads import BBoxHeadAVA
    from .roi_extractors import SingleRoIExtractor3D
    from .roi_head import AVARoIHead
    from .shared_heads import ACRNHead, FBOHead, LFBInferHead

    for module in [
            AVARoIHead, BBoxHeadAVA, SingleRoIExtractor3D, ACRNHead, FBOHead,
            LFBInferHead
    ]:

        MMDET_MODELS.register_module()(module)

    __all__ = [
        'AVARoIHead', 'BBoxHeadAVA', 'SingleRoIExtractor3D', 'ACRNHead',
        'FBOHead', 'LFBInferHead'
    ]

except (ImportError, ModuleNotFoundError):
    pass
