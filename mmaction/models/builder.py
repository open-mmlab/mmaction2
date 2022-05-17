# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmengine.registry import MODELS as MMEngine_MODELS
from mmengine.registry import Registry

MODELS = Registry('models', parent=MMEngine_MODELS)
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
RECOGNIZERS = MODELS
LOSSES = MODELS
LOCALIZERS = MODELS

try:
    from mmdet.models.builder import DETECTORS, build_detector
except (ImportError, ModuleNotFoundError):
    # Define an empty registry and building func, so that can import
    DETECTORS = MODELS

    def build_detector(cfg, train_cfg, test_cfg):
        warnings.warn(
            'Failed to import `DETECTORS`, `build_detector` from '
            '`mmdet.models.builder`. You will be unable to register or build '
            'a spatio-temporal detection model. ')


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_recognizer(cfg):
    """Build recognizer."""
    return RECOGNIZERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_localizer(cfg):
    """Build localizer."""
    return LOCALIZERS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)
