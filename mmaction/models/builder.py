import warnings

import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

from mmaction.utils import import_module_error_func
from .registry import BACKBONES, HEADS, LOCALIZERS, LOSSES, NECKS, RECOGNIZERS

try:
    from mmdet.models.builder import DETECTORS, build_detector
except (ImportError, ModuleNotFoundError):
    # Define an empty registry and building func, so that can import
    DETECTORS = Registry('detector')

    @import_module_error_func('mmdet')
    def build_detector(cfg, train_cfg, test_cfg):
        pass


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, it is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model. Details see this '
            'PR: https://github.com/open-mmlab/mmaction2/pull/629',
            UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, RECOGNIZERS,
                 dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_localizer(cfg):
    """Build localizer."""
    return build(cfg, LOCALIZERS)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in LOCALIZERS:
        return build_localizer(cfg)
    if obj_type in RECOGNIZERS:
        return build_recognizer(cfg, train_cfg, test_cfg)
    if obj_type in DETECTORS:
        if train_cfg is not None or test_cfg is not None:
            warnings.warn(
                'train_cfg and test_cfg is deprecated, '
                'please specify them in model. Details see this '
                'PR: https://github.com/open-mmlab/mmaction2/pull/629',
                UserWarning)
        return build_detector(cfg, train_cfg, test_cfg)
    model_in_mmdet = ['FastRCNN']
    if obj_type in model_in_mmdet:
        raise ImportError(
            'Please install mmdet for spatial temporal detection tasks.')
    raise ValueError(f'{obj_type} is not registered in '
                     'LOCALIZERS, RECOGNIZERS or DETECTORS')


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)
