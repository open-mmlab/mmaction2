# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import build_recognizer
from ..base import generate_recognizer_demo_inputs, get_recognizer_cfg


def test_tsn():
    config = get_recognizer_cfg('tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

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

    # test forward dummy
    recognizer.forward_dummy(imgs, softmax=False)
    res = recognizer.forward_dummy(imgs, softmax=True)[0]
    assert torch.min(res) >= 0
    assert torch.max(res) <= 1

    mmcls_backbone = dict(
        type='mmcls.ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=4,
        style='pytorch')
    config.model['backbone'] = mmcls_backbone

    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test mixup forward
    config = get_recognizer_cfg(
        'tsn/tsn_r50_video_mixup_1x1x8_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)
    input_shape = (2, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']
    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # test torchvision backbones
    tv_backbone = dict(type='torchvision.densenet161', pretrained=True)
    config.model['backbone'] = tv_backbone
    config.model['cls_head']['in_channels'] = 2208

    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)


def test_tsm():
    config = get_recognizer_cfg('tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test twice sample + 3 crops
    input_shape = (2, 48, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    config.model.test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(config.model)

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
    config = get_recognizer_cfg('trn/trn_r50_1x1x8_50e_sthv1_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test twice sample + 3 crops
    input_shape = (2, 48, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    config.model.test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(config.model)

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
    config = get_recognizer_cfg('tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 8, 3, 224, 224)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)
    assert 'loss_aux' in losses and 'loss_cls' in losses

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
    with torch.no_grad():
        _recognizer = build_recognizer(config.model)
        img_list = [img[None, :] for img in imgs]
        if hasattr(_recognizer, 'forward_dummy'):
            _recognizer.forward = _recognizer.forward_dummy
        for one_img in img_list:
            _recognizer(one_img)


def test_tanet():
    config = get_recognizer_cfg(
        'tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)

    # test twice sample + 3 crops
    input_shape = (2, 48, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    config.model.test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(config.model)

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
    config = get_recognizer_cfg('tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained'] = None
    timm_backbone = dict(type='timm.efficientnet_b0', pretrained=False)
    config.model['backbone'] = timm_backbone
    config.model['cls_head']['in_channels'] = 1280

    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape)

    imgs = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [img[None, :] for img in imgs]
        for one_img in img_list:
            recognizer(one_img, None, return_loss=False)
