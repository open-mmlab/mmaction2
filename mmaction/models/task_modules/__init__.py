# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS

    from .assigners import MaxIoUAssignerAVA

    MMDET_TASK_UTILS.register_module()(MaxIoUAssignerAVA)

    __all__ = ['MaxIoUAssignerAVA']

except (ImportError, ModuleNotFoundError):
    pass
