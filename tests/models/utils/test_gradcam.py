# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_recognizer_cfg
from mmaction.utils import register_all_modules
from mmaction.utils.gradcam_utils import GradCAM

register_all_modules()


def _get_target_shapes(input_shape, num_classes=400, model_type='2D'):
    if model_type not in ['2D', '3D']:
        raise ValueError(f'Data type {model_type} is not available')

    preds_target_shape = (input_shape[0], num_classes)
    if model_type == '3D':
        # input shape (batch_size, num_crops*num_clips, C, clip_len, H, W)
        # target shape (batch_size*num_crops*num_clips, clip_len, H, W, C)
        blended_imgs_target_shape = (input_shape[0] * input_shape[1],
                                     input_shape[3], input_shape[4],
                                     input_shape[5], input_shape[2])
    else:
        # input shape (batch_size, num_segments, C, H, W)
        # target shape (batch_size, num_segments, H, W, C)
        blended_imgs_target_shape = (input_shape[0], input_shape[1],
                                     input_shape[3], input_shape[4],
                                     input_shape[2])

    return blended_imgs_target_shape, preds_target_shape


def _do_test_2D_models(recognizer,
                       target_layer_name,
                       input_shape,
                       num_classes=400,
                       device='cpu'):
    demo_data = {
        'inputs': [torch.randint(0, 256, input_shape[1:])],
        'data_samples': [ActionDataSample().set_gt_label(2)]
    }

    recognizer = recognizer.to(device)
    gradcam = GradCAM(recognizer, target_layer_name)

    blended_imgs_target_shape, preds_target_shape = _get_target_shapes(
        input_shape, num_classes=num_classes, model_type='2D')

    blended_imgs, preds = gradcam(demo_data)
    assert blended_imgs.size() == blended_imgs_target_shape
    assert preds.size() == preds_target_shape

    blended_imgs, preds = gradcam(demo_data, True)
    assert blended_imgs.size() == blended_imgs_target_shape
    assert preds.size() == preds_target_shape


def _do_test_3D_models(recognizer,
                       target_layer_name,
                       input_shape,
                       num_classes=400):
    blended_imgs_target_shape, preds_target_shape = _get_target_shapes(
        input_shape, num_classes=num_classes, model_type='3D')
    demo_data = {
        'inputs': [torch.randint(0, 256, input_shape[1:])],
        'data_samples': [ActionDataSample().set_gt_label(2)]
    }

    gradcam = GradCAM(recognizer, target_layer_name)

    blended_imgs, preds = gradcam(demo_data)
    assert blended_imgs.size() == blended_imgs_target_shape
    assert preds.size() == preds_target_shape

    blended_imgs, preds = gradcam(demo_data, True)
    assert blended_imgs.size() == blended_imgs_target_shape
    assert preds.size() == preds_target_shape


def test_tsn():
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config

    input_shape = (1, 25, 3, 32, 32)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_2D_models(recognizer, target_layer_name, input_shape)


def test_i3d():
    config = get_recognizer_cfg(
        'i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)
    recognizer.cfg = config

    input_shape = (1, 1, 3, 32, 32, 32)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_r2plus1d():
    config = get_recognizer_cfg(
        'r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['norm_cfg'] = dict(type='BN3d')

    recognizer = MODELS.build(config.model)
    recognizer.cfg = config

    input_shape = (1, 3, 3, 8, 16, 16)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_slowfast():
    config = get_recognizer_cfg(
        'slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py')

    recognizer = MODELS.build(config.model)
    recognizer.cfg = config

    input_shape = (1, 1, 3, 32, 32, 32)
    target_layer_name = 'backbone/slow_path/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_tsm():
    config = get_recognizer_cfg(
        'tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    target_layer_name = 'backbone/layer4/1/relu'

    # base config
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 8, 3, 32, 32)
    _do_test_2D_models(recognizer, target_layer_name, input_shape)

    # test twice sample + 3 crops, 2*3*8=48
    config.model.test_cfg = dict(average_clips='prob')
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 48, 3, 32, 32)
    _do_test_2D_models(recognizer, target_layer_name, input_shape)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_csn():
    config = get_recognizer_cfg(
        'csn/ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py'  # noqa: E501
    )
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 1, 3, 32, 16, 16)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_tpn():
    target_layer_name = 'backbone/layer4/1/relu'

    config = get_recognizer_cfg(
        'tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['num_segments'] = 4
    config.model.test_cfg['fcn_test'] = False
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config

    input_shape = (1, 4, 3, 16, 16)
    _do_test_2D_models(recognizer, target_layer_name, input_shape, 174)

    config = get_recognizer_cfg(
        'tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model.test_cfg['fcn_test'] = False
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 3, 3, 4, 16, 16)
    _do_test_3D_models(recognizer, target_layer_name, input_shape)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_c3d():
    config = get_recognizer_cfg(
        'c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 1, 3, 16, 112, 112)
    target_layer_name = 'backbone/conv5a/activate'
    _do_test_3D_models(recognizer, target_layer_name, input_shape, 101)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_tin():
    config = get_recognizer_cfg(
        'tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    target_layer_name = 'backbone/layer4/1/relu'

    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 8, 3, 64, 64)
    _do_test_2D_models(
        recognizer, target_layer_name, input_shape, device='cuda:0')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_x3d():
    config = get_recognizer_cfg('x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = MODELS.build(config.model)
    recognizer.cfg = config
    input_shape = (1, 1, 3, 13, 16, 16)
    target_layer_name = 'backbone/layer4/1/relu'
    _do_test_3D_models(recognizer, target_layer_name, input_shape)
