# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import MagicMock

import pytest
import torch
from mmengine.utils import digit_version

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_recognizer_cfg
from mmaction.utils import register_all_modules


def train_test_step(cfg, input_shape):
    recognizer = MODELS.build(cfg.model)
    num_classes = cfg.model.cls_head.num_classes
    batch_size = input_shape[0]
    input_shape = input_shape[1:]
    data_batch = {
        'inputs':
        [torch.randint(0, 256, input_shape) for i in range(batch_size)],
        'data_samples':
        [ActionDataSample().set_gt_label(2) for i in range(batch_size)]
    }

    # test train_step
    optim_wrapper = MagicMock()
    loss_vars = recognizer.train_step(data_batch, optim_wrapper)
    assert 'loss' in loss_vars
    assert 'loss_cls' in loss_vars
    optim_wrapper.update_params.assert_called_once()

    # test test_step
    with torch.no_grad():
        predictions = recognizer.test_step(data_batch)
    score = predictions[0].pred_score
    assert len(predictions) == batch_size
    assert score.shape == torch.Size([num_classes])
    assert torch.min(score) >= 0
    assert torch.max(score) <= 1

    # test twice sample + 3 crops
    num_views = input_shape[0] * 2 * 3
    input_shape = (num_views, *input_shape[1:])
    data_batch['inputs'] = [torch.randint(0, 256, input_shape)]
    with torch.no_grad():
        predictions = recognizer.test_step(data_batch)
    score = predictions[0].pred_score
    assert len(predictions) == batch_size
    assert score.shape == torch.Size([num_classes])

    return loss_vars, predictions


def test_tsn():
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None

    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)


def test_tsn_mmcls_backbone():
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    # test mmcls backbone
    mmcls_backbone = dict(
        type='mmcls.ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=4,
        style='pytorch')
    config.model['backbone'] = mmcls_backbone

    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)

    from mmcls.models import ResNeXt
    mmcls_backbone['type'] = ResNeXt
    config.model['backbone'] = mmcls_backbone

    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)


def test_tsn_mobileone():
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/custom_backbones/tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb.py'  # noqa: E501
    )
    config.model['backbone']['init_cfg'] = None
    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)


def test_tsn_timm_backbone():
    # test tsn from timm
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    timm_backbone = dict(type='timm.efficientnet_b0', pretrained=False)
    config.model['backbone'] = timm_backbone
    config.model['cls_head']['in_channels'] = 1280

    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)
    import timm
    if digit_version(timm.__version__) <= digit_version('0.6.7'):
        feature_shape = 'NLC'
    else:
        feature_shape = 'NHWC'

    timm_swin = dict(
        type='timm.swin_base_patch4_window7_224',
        pretrained=False,
        feature_shape=feature_shape)
    config.model['backbone'] = timm_swin
    config.model['cls_head']['in_channels'] = 1024

    input_shape = (1, 3, 3, 224, 224)
    train_test_step(config, input_shape)


def test_tsn_tv_backbone():
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    # test tv backbone
    tv_backbone = dict(type='torchvision.densenet161', pretrained=True)
    config.model['backbone'] = tv_backbone
    config.model['cls_head']['in_channels'] = 2208

    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)

    from torchvision.models import densenet161
    tv_backbone = dict(type=densenet161, pretrained=True)
    config.model['backbone'] = tv_backbone
    config.model['cls_head']['in_channels'] = 2208

    input_shape = (1, 3, 3, 32, 32)
    train_test_step(config, input_shape)


def test_tsm():
    register_all_modules()
    # test tsm-mobilenetv2
    config = get_recognizer_cfg(
        'tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py'  # noqa: E501
    )
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['pretrained2d'] = None

    input_shape = (1, 8, 3, 32, 32)
    train_test_step(config, input_shape)

    # test tsm-res50
    config = get_recognizer_cfg(
        'tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['pretrained2d'] = None

    input_shape = (1, 8, 3, 32, 32)
    train_test_step(config, input_shape)

    # test tsm-mobileone
    config = get_recognizer_cfg(
        'tsm/tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb.py'  # noqa: E501
    )
    config.model['backbone']['init_cfg'] = None
    config.model['backbone']['pretrained2d'] = None

    input_shape = (1, 16, 3, 32, 32)
    train_test_step(config, input_shape)


def test_trn():
    register_all_modules()
    config = get_recognizer_cfg(
        'trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py')
    config.model['backbone']['pretrained'] = None

    input_shape = (1, 8, 3, 32, 32)
    train_test_step(config, input_shape)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_tpn():
    register_all_modules()
    config = get_recognizer_cfg(
        'tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.py')
    config.model['backbone']['pretrained'] = None

    input_shape = (1, 8, 3, 64, 64)
    train_test_step(config, input_shape)


def test_tanet():
    register_all_modules()
    config = get_recognizer_cfg('tanet/tanet_imagenet-pretrained-r50_8xb8-'
                                'dense-1x1x8-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None

    input_shape = (1, 8, 3, 32, 32)
    train_test_step(config, input_shape)
