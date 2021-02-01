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
        return build_detector(cfg, train_cfg, test_cfg)
    raise ValueError(f'{obj_type} is not registered in '
                     'LOCALIZERS, RECOGNIZERS or DETECTORS')


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)
