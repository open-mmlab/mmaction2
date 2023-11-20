from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.runtime_info_hook import RuntimeInfoHook
from mmengine.hooks.sampler_seed_hook import DistSamplerSeedHook
from mmengine.hooks.sync_buffer_hook import SyncBuffersHook
from mmengine.runner.log_processor import LogProcessor

from mmaction.visualization.action_visualizer import ActionVisualizer
from mmaction.visualization.video_backend import LocalVisBackend

# hooks
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
