# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from ..base import generate_recognizer_demo_inputs, get_recognizer_cfg


def assert_output(x):
    if not isinstance(x, torch.Tensor):
        assert isinstance(x, tuple)
        for i in x:
            assert isinstance(i, torch.Tensor)


def test_i3d():
    register_all_modules()
    config = get_recognizer_cfg(
        'i3d/i3d_r50_8xb8-32x2x1-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 8, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            recognizer = recognizer.cuda()
            imgs = imgs.cuda()
            gt_labels = gt_labels.cuda()
            losses = recognizer(imgs, gt_labels)
            assert isinstance(losses, dict)

            # Test forward test
            with torch.no_grad():
                img_list = [img[None, :] for img in imgs]
                for one_img in img_list:
                    recognizer(one_img, None, return_loss=False)

            # Test forward gradcam
            recognizer(imgs, gradcam=True)
            for one_img in img_list:
                recognizer(one_img, gradcam=True)

            # Test forward dummy
            recognizer.forward_dummy(imgs, softmax=False)
            res = recognizer.forward_dummy(imgs, softmax=True)[0]
            assert torch.min(res) >= 0
            assert torch.max(res) <= 1

    else:
        losses = recognizer(imgs, gt_labels)
        assert_output(losses)

        # Test forward test
        with torch.no_grad():
            img_list = [img[None, :] for img in imgs]
            for one_img in img_list:
                recognizer(one_img, None, return_loss=False)

        # Test forward gradcam
        recognizer(imgs, gradcam=True)
        for one_img in img_list:
            recognizer(one_img, gradcam=True)


def test_r2plus1d():
    register_all_modules()
    config = get_recognizer_cfg(
        'r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['norm_cfg'] = dict(type='BN3d')

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 8, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            recognizer = recognizer.cuda()
            imgs = imgs.cuda()
            gt_labels = gt_labels.cuda()
            losses = recognizer(imgs, gt_labels)
            assert isinstance(losses, dict)

            # Test forward test
            with torch.no_grad():
                img_list = [img[None, :] for img in imgs]
                for one_img in img_list:
                    recognizer(one_img, None, return_loss=False)

            # Test forward gradcam
            recognizer(imgs, gradcam=True)
            for one_img in img_list:
                recognizer(one_img, gradcam=True)
    else:
        losses = recognizer(imgs, gt_labels)
        assert_output(losses)

        # Test forward test
        with torch.no_grad():
            img_list = [img[None, :] for img in imgs]
            for one_img in img_list:
                recognizer(one_img, None, return_loss=False)

        # Test forward gradcam
        recognizer(imgs, gradcam=True)
        for one_img in img_list:
            recognizer(one_img, gradcam=True)


def test_slowfast():
    register_all_modules()
    config = get_recognizer_cfg(
        'slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py')

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 16, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            recognizer = recognizer.cuda()
            imgs = imgs.cuda()
            gt_labels = gt_labels.cuda()
            losses = recognizer(imgs, gt_labels)
            assert isinstance(losses, dict)

            # Test forward test
            with torch.no_grad():
                img_list = [img[None, :] for img in imgs]
                for one_img in img_list:
                    recognizer(one_img, None, return_loss=False)

            # Test forward gradcam
            recognizer(imgs, gradcam=True)
            for one_img in img_list:
                recognizer(one_img, gradcam=True)
    else:
        losses = recognizer(imgs, gt_labels)
        assert_output(losses)

        # Test forward test
        with torch.no_grad():
            img_list = [img[None, :] for img in imgs]
            for one_img in img_list:
                recognizer(one_img, None, return_loss=False)

        # Test forward gradcam
        recognizer(imgs, gradcam=True)
        for one_img in img_list:
            recognizer(one_img, gradcam=True)

        # Test the feature max_testing_views
        if config.model.test_cfg is None:
            config.model.test_cfg = {'max_testing_views': 1}
        else:
            config.model.test_cfg['max_testing_views'] = 1
        recognizer = MODELS.build(config.model)
        with torch.no_grad():
            img_list = [img[None, :] for img in imgs]
            for one_img in img_list:
                recognizer(one_img, None, return_loss=False)


def test_csn():
    register_all_modules()
    config = get_recognizer_cfg(
        'csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 8, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    # parrots 3dconv is only implemented on gpu
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            recognizer = recognizer.cuda()
            imgs = imgs.cuda()
            gt_labels = gt_labels.cuda()
            losses = recognizer(imgs, gt_labels)
            assert isinstance(losses, dict)

            # Test forward test
            with torch.no_grad():
                img_list = [img[None, :] for img in imgs]
                for one_img in img_list:
                    recognizer(one_img, None, return_loss=False)

            # Test forward gradcam
            recognizer(imgs, gradcam=True)
            for one_img in img_list:
                recognizer(one_img, gradcam=True)
    else:
        losses = recognizer(imgs, gt_labels)
        assert_output(losses)

        # Test forward test
        with torch.no_grad():
            img_list = [img[None, :] for img in imgs]
            for one_img in img_list:
                recognizer(one_img, None, return_loss=False)

        # Test forward gradcam
        recognizer(imgs, gradcam=True)
        for one_img in img_list:
            recognizer(one_img, gradcam=True)


def test_tpn():
    register_all_modules()
    config = get_recognizer_cfg(
        'tpn/tpn_slowonly_r50_8x8x1_150e_8xb8_kinetics_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 8, 3, 1, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert_output(losses)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)


def test_timesformer():
    register_all_modules()
    config = get_recognizer_cfg(
        'timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['img_size'] = 32

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 8, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert_output(losses)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)


def test_c3d():
    register_all_modules()
    config = get_recognizer_cfg(
        'c3d/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb.py')
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['out_dim'] = 512

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 16, 28, 28)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert_output(losses)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)
