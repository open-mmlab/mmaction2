import pytest
import torch

from mmaction.models import build_recognizer
from mmaction.utils.gradcam_utils import GradCAM
from .base import generate_gradcam_inputs, get_recognizer_cfg


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
    demo_inputs = generate_gradcam_inputs(input_shape)
    demo_inputs['imgs'] = demo_inputs['imgs'].to(device)
    demo_inputs['label'] = demo_inputs['label'].to(device)

    recognizer = recognizer.to(device)
    gradcam = GradCAM(recognizer, target_layer_name)

    blended_imgs_target_shape, preds_target_shape = _get_target_shapes(
        input_shape, num_classes=num_classes, model_type='2D')

    blended_imgs, preds = gradcam(demo_inputs)
    assert blended_imgs.size() == blended_imgs_target_shape
    assert preds.size() == preds_target_shape

    blended_imgs, preds = gradcam(demo_inputs, True)
    assert blended_imgs.size() == blended_imgs_target_shape
    assert preds.size() == preds_target_shape


def _do_test_3D_models(recognizer,
                       target_layer_name,
                       input_shape,
                       num_classes=400):
    blended_imgs_target_shape, preds_target_shape = _get_target_shapes(
        input_shape, num_classes=num_classes, model_type='3D')
    demo_inputs = generate_gradcam_inputs(input_shape, '3D')

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            recognizer = recognizer.cuda()
            demo_inputs['imgs'] = demo_inputs['imgs'].cuda()
            demo_inputs['label'] = demo_inputs['label'].cuda()
            gradcam = GradCAM(recognizer, target_layer_name)

            blended_imgs, preds = gradcam(demo_inputs)
            assert blended_imgs.size() == blended_imgs_target_shape
            assert preds.size() == preds_target_shape

            blended_imgs, preds = gradcam(demo_inputs, True)
            assert blended_imgs.size() == blended_imgs_target_shape
            assert preds.size() == preds_target_shape
    else:
        gradcam = GradCAM(recognizer, target_layer_name)

        blended_imgs, preds = gradcam(demo_inputs)
        assert blended_imgs.size() == blended_imgs_target_shape
        assert preds.size() == preds_target_shape

        blended_imgs, preds = gradcam(demo_inputs, True)
        assert blended_imgs.size() == blended_imgs_target_shape
        assert preds.size() == preds_target_shape


def test_tsn():
    config = get_recognizer_cfg('tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config

    input_shape = (1, 25, 3, 32, 32)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_2D_models(recognizer, target_layer_name, input_shape)


def test_i3d():
    config = get_recognizer_cfg('i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)
    recognizer.cfg = config

    input_shape = [1, 1, 3, 32, 32, 32]
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_r2plus1d():
    config = get_recognizer_cfg(
        'r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['norm_cfg'] = dict(type='BN3d')

    recognizer = build_recognizer(config.model)
    recognizer.cfg = config

    input_shape = (1, 3, 3, 8, 32, 32)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_slowfast():
    config = get_recognizer_cfg(
        'slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py')

    recognizer = build_recognizer(config.model)
    recognizer.cfg = config

    input_shape = (1, 1, 3, 32, 32, 32)
    target_layer_name = 'backbone/slow_path/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_tsm():
    config = get_recognizer_cfg('tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    target_layer_name = 'backbone/layer4/1/relu'

    # base config
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 8, 3, 32, 32)
    _do_test_2D_models(recognizer, target_layer_name, input_shape)

    # test twice sample + 3 crops, 2*3*8=48
    config.model.test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 48, 3, 32, 32)
    _do_test_2D_models(recognizer, target_layer_name, input_shape)


def test_csn():
    config = get_recognizer_cfg(
        'csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 1, 3, 32, 32, 32)
    target_layer_name = 'backbone/layer4/1/relu'

    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_tpn():
    target_layer_name = 'backbone/layer4/1/relu'

    config = get_recognizer_cfg('tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config

    input_shape = (1, 8, 3, 32, 32)
    _do_test_2D_models(recognizer, target_layer_name, input_shape, 174)

    config = get_recognizer_cfg(
        'tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 3, 3, 8, 32, 32)
    _do_test_3D_models(recognizer, target_layer_name, input_shape)


def test_c3d():
    config = get_recognizer_cfg('c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 1, 3, 16, 112, 112)
    target_layer_name = 'backbone/conv5a/activate'
    _do_test_3D_models(recognizer, target_layer_name, input_shape, 101)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_tin():
    config = get_recognizer_cfg(
        'tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    target_layer_name = 'backbone/layer4/1/relu'

    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 8, 3, 64, 64)
    _do_test_2D_models(
        recognizer, target_layer_name, input_shape, device='cuda:0')


def test_x3d():
    config = get_recognizer_cfg('x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)
    recognizer.cfg = config
    input_shape = (1, 1, 3, 13, 32, 32)
    target_layer_name = 'backbone/layer4/1/relu'
    _do_test_3D_models(recognizer, target_layer_name, input_shape)
