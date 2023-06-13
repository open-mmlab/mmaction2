# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmaction.registry import MODELS
from mmaction.testing import (generate_recognizer_demo_inputs,
                              get_recognizer_cfg)
from mmaction.utils import register_all_modules


def test_tsn():
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)
    """
    TODO
    mmcls_backbone = dict(
        type='mmcls.ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=4,
        style='pytorch')
    config.model['backbone'] = mmcls_backbone

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)


    # test mixup forward
    # TODO
    config = get_recognizer_cfg(
        'tsn/tsn_r50_video_mixup_1x1x8_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = MODELS.build(config.model)
    input_shape = (2, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']
    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # test torchvision backbones
    # TODO
    tv_backbone = dict(type='torchvision.densenet161', pretrained=True)
    config.model['backbone'] = tv_backbone
    config.model['cls_head']['in_channels'] = 2208

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)
    """


def test_tsm():
    register_all_modules()
    config = get_recognizer_cfg(
        'tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-50e_kinetics400-rgb.py'  # noqa: E501
    )
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)
    recognizer.init_weights()

    config = get_recognizer_cfg(
        'tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)
    recognizer.init_weights()

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test twice sample + 3 crops
    input_shape = (2, 48, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)


def test_trn():
    register_all_modules()
    config = get_recognizer_cfg(
        'trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test twice sample + 3 crops
    input_shape = (2, 48, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_tpn():
    register_all_modules()
    config = get_recognizer_cfg(
        'tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)

    if not isinstance(losses, torch.Tensor):
        for i in losses:
            assert isinstance(i, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)


def test_tanet():
    register_all_modules()
    config = get_recognizer_cfg('tanet/tanet_imagenet-pretrained-r50_8xb8-'
                                'dense-1x1x8-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = MODELS.build(config.model)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test twice sample + 3 crops
    input_shape = (2, 48, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # Test forward gradcam
    recognizer(imgs, gradcam=True)
    for one_img in img_list:
        recognizer(one_img, gradcam=True)


def test_timm_backbone():
    # test tsn from timm
    register_all_modules()
    config = get_recognizer_cfg(
        'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    config.model['backbone']['pretrained'] = None
    timm_backbone = dict(type='timm.efficientnet_b0', pretrained=False)
    config.model['backbone'] = timm_backbone
    config.model['cls_head']['in_channels'] = 1280

    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)
