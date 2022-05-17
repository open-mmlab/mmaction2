# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import DefaultScope
from mmengine.registry import METRICS as MMEngine_METRICS
from mmengine.registry import Registry

# default_scope = DefaultScope.get_instance('mmengine', scope_name='mmaction')

METRICS = Registry('metrics', parent=MMEngine_METRICS)
