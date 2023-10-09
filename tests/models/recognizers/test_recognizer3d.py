# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_recognizer_cfg
from mmaction.utils import register_all_modules


def train_test_step(cfg, input_shape):
    recognizer = MODELS.build(cfg.model)
    num_classes = cfg.model.cls_head.num_classes
    data_batch = {
        'inputs': [torch.randint(0, 256, input_shape)],
        'data_samples': [ActionDataSample().set_gt_label(2)]
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
    assert len(predictions) == 1
    assert score.shape == torch.Size([num_classes])
    assert torch.min(score) >= 0
    assert torch.max(score) <= 1

    # test when average_clips is None
    recognizer.cls_head.average_clips = None
    num_views = 3
    input_shape = (num_views, *input_shape[1:])
    data_batch['inputs'] = [torch.randint(0, 256, input_shape)]
    with torch.no_grad():
        predictions = recognizer.test_step(data_batch)
    score = predictions[0].pred_score
    assert len(predictions) == 1
    assert score.shape == torch.Size([num_views, num_classes])

    return loss_vars, predictions


def test_i3d():
    register_all_modules()
    config = get_recognizer_cfg(
        'i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    input_shape = (1, 3, 8, 64, 64)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_r2plus1d():
    register_all_modules()
    config = get_recognizer_cfg(
        'r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['norm_cfg'] = dict(type='BN3d')
    input_shape = (1, 3, 8, 64, 64)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_slowfast():
    register_all_modules()
    config = get_recognizer_cfg(
        'slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py')
    input_shape = (1, 3, 16, 64, 64)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_csn():
    register_all_modules()
    config = get_recognizer_cfg(
        'csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    input_shape = (1, 3, 8, 64, 64)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_timesformer():
    register_all_modules()
    config = get_recognizer_cfg(
        'timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['img_size'] = 32
    input_shape = (1, 3, 8, 32, 32)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_c3d():
    register_all_modules()
    config = get_recognizer_cfg(
        'c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['out_dim'] = 512
    input_shape = (1, 3, 16, 28, 28)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_slowonly():
    register_all_modules()
    config = get_recognizer_cfg(
        'slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    input_shape = (1, 3, 4, 32, 32)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_tpn_slowonly():
    register_all_modules()
    config = get_recognizer_cfg('tpn/tpn-slowonly_imagenet-pretrained-r50_'
                                '8xb8-8x8x1-150e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    input_shape = (1, 3, 4, 48, 48)  # M C T H W
    loss_vars, _ = train_test_step(config, input_shape=input_shape)
    assert 'loss_aux' in loss_vars
    assert loss_vars['loss_cls'] + loss_vars['loss_aux'] == loss_vars['loss']


def test_swin():
    register_all_modules()
    config = get_recognizer_cfg('swin/swin-tiny-p244-w877_in1k-pre_'
                                '8xb8-amp-32x2x1-30e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    input_shape = (1, 3, 4, 64, 64)  # M C T H W
    train_test_step(config, input_shape=input_shape)


def test_c2d():
    register_all_modules()
    config = get_recognizer_cfg(
        'c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    input_shape = (1, 3, 8, 64, 64)  # M C T H W
    train_test_step(config, input_shape=input_shape)
