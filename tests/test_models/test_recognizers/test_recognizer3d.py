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


def test_i3d():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py')
    model['backbone']['pretrained2d'] = False
    model['backbone']['pretrained'] = None

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


def test_slowfast():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py')

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


def test_csn():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py')
    model['backbone']['pretrained2d'] = False
    model['backbone']['pretrained'] = None

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


def test_c3d():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py')
    model['backbone']['pretrained'] = None

    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 3, 16, 112, 112)
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
