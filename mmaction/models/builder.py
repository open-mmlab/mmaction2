import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from mmaction.utils import import_module_error_func

MODELS = Registry('models', parent=MMCV_MODELS)
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

    @import_module_error_func('mmdet')
    def build_detector(cfg, train_cfg, test_cfg):
        pass


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model. Details see this '
            'PR: https://github.com/open-mmlab/mmaction2/pull/629',
            UserWarning)
    assert cfg.get(
        'train_cfg'
    ) is None or train_cfg is None, 'train_cfg specified in both outer field and model field'  # noqa: E501
    assert cfg.get(
        'test_cfg'
    ) is None or test_cfg is None, 'test_cfg specified in both outer field and model field '  # noqa: E501
    return RECOGNIZERS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_localizer(cfg):
    """Build localizer."""
    return LOCALIZERS.build(cfg)


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
    return NECKS.build(cfg)
