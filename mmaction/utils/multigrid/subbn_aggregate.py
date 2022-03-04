# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, Hook


def aggregate_sub_bn_status(module):
    from mmaction.models import SubBatchNorm3D
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3D):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_status(child)
    return count


@HOOKS.register_module()
class SubBatchNorm3dAggregationHook(Hook):
    """Recursively find all SubBN modules and aggregate sub-BN stats."""

    def after_train_epoch(self, runner):
        _ = aggregate_sub_bn_status(runner.model)
