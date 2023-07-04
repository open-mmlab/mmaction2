# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook, RuntimeInfoHook,
                            SyncBuffersHook)
from mmengine.runner import LogProcessor

from mmaction.visualization import ActionVisualizer, LocalVisBackend

default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type=RuntimeInfoHook),
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=20, ignore_last=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=1, save_best='auto'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    sync_buffers=dict(type=SyncBuffersHook))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type=LogProcessor, window_size=20, by_epoch=True)

vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(type=ActionVisualizer, vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
