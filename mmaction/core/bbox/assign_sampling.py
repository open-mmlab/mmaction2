import mmcv

from . import assigners, samplers


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, assigners, default_args=kwargs)
    raise TypeError(f'Invalid type {type(cfg)} for building a sampler')


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, samplers, default_args=kwargs)
    raise TypeError(f'Invalid type {type(cfg)} for building a sampler')
