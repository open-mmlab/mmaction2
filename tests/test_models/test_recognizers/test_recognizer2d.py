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
