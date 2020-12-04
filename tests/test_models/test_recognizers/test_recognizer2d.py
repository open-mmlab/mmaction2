import os.path as osp

import mmcv
import torch
from tests.test_models.test_common_modules.test_base_recognizer import \
    generate_demo_inputs

from mmaction.models import build_recognizer


def _get_recognizer_cfg(fname):
    """Grab configs necessary to create a recognizer.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    repo_dpath = osp.dirname(osp.dirname(osp.dirname(__file__)))
    config_dpath = osp.join(repo_dpath, 'configs/recognition')
    config_fpath = osp.join(config_dpath, fname)
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config = mmcv.Config.fromfile(config_fpath)
    return config.model, config.train_cfg, config.test_cfg


def test_tsn():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
    model['backbone']['pretrained'] = None

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 3, 32, 32)
    demo_inputs = generate_demo_inputs(input_shape)

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


def test_r2plus1d():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py')
    model['backbone']['pretrained2d'] = False
    model['backbone']['pretrained'] = None
    model['backbone']['norm_cfg'] = dict(type='BN3d')

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 3, 8, 32, 32)
    demo_inputs = generate_demo_inputs(input_shape, '3D')

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
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py')
    model['backbone']['pretrained'] = None

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 8, 3, 32, 32)
    demo_inputs = generate_demo_inputs(input_shape)

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
    demo_inputs = generate_demo_inputs(input_shape)
    imgs = demo_inputs['imgs']

    test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

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
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py')
    model['backbone']['pretrained'] = None

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 8, 3, 224, 224)
    demo_inputs = generate_demo_inputs(input_shape)

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

    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py')
    model['backbone']['pretrained'] = None

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 8, 3, 1, 224, 224)
    demo_inputs = generate_demo_inputs(input_shape, '3D')

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


def _get_audio_recognizer_cfg(fname):
    """Grab configs necessary to create a audio recognizer.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    repo_dpath = osp.dirname(osp.dirname(osp.dirname(__file__)))
    config_dpath = osp.join(repo_dpath, 'configs/recognition_audio/')
    config_fpath = osp.join(config_dpath, fname)
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config = mmcv.Config.fromfile(config_fpath)
    return config.model, config.train_cfg, config.test_cfg


def test_audio_recognizer():
    model, train_cfg, test_cfg = _get_audio_recognizer_cfg(
        'resnet/tsn_r18_64x1x1_100e_kinetics400_audio_feature.py')
    model['backbone']['pretrained'] = None

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 1, 128, 80)
    demo_inputs = generate_demo_inputs(input_shape, model_type='audio')

    audios = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(audios, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        audio_list = [audio[None, :] for audio in audios]
        for one_spectro in audio_list:
            recognizer(one_spectro, None, return_loss=False)
