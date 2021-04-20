import torch

from mmaction.models import build_recognizer
from ..base import generate_recognizer_demo_inputs, get_recognizer_cfg


def test_i3d():
    config = get_recognizer_cfg('i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

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


def test_r2plus1d():
    config = get_recognizer_cfg(
        'r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None
    config.model['backbone']['norm_cfg'] = dict(type='BN3d')

    recognizer = build_recognizer(config.model)

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
    config = get_recognizer_cfg(
        'slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py')

    recognizer = build_recognizer(config.model)

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

        # Test the feature max_testing_views
        config.model.test_cfg['max_testing_views'] = 1
        recognizer = build_recognizer(config.model)
        with torch.no_grad():
            img_list = [img[None, :] for img in imgs]
            for one_img in img_list:
                recognizer(one_img, None, return_loss=False)


def test_csn():
    config = get_recognizer_cfg(
        'csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py')
    config.model['backbone']['pretrained2d'] = False
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

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


def test_tpn():
    config = get_recognizer_cfg(
        'tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 8, 3, 1, 32, 32)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

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

    # Test dummy forward
    with torch.no_grad():
        _recognizer = build_recognizer(config.model)
        img_list = [img[None, :] for img in imgs]
        if hasattr(_recognizer, 'forward_dummy'):
            _recognizer.forward = _recognizer.forward_dummy
        for one_img in img_list:
            _recognizer(one_img)


def test_c3d():
    config = get_recognizer_cfg('c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 3, 16, 112, 112)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, '3D')

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
