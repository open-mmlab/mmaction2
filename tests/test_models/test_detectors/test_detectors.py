# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from ..base import generate_detector_demo_inputs, get_detector_cfg

try:
    from mmaction.models import build_detector
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@pytest.mark.skipif(not mmdet_imported, reason='requires mmdet')
def test_ava_detector():
    config = get_detector_cfg('ava/slowonly_kinetics_pretrained_r50_'
                              '4x16x1_20e_ava_rgb.py')
    detector = build_detector(config.model)

    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            train_demo_inputs = generate_detector_demo_inputs(
                train=True, device='cuda')
            test_demo_inputs = generate_detector_demo_inputs(
                train=False, device='cuda')
            detector = detector.cuda()

            losses = detector(**train_demo_inputs)
            assert isinstance(losses, dict)

            # Test forward test
            with torch.no_grad():
                _ = detector(**test_demo_inputs, return_loss=False)
    else:
        train_demo_inputs = generate_detector_demo_inputs(train=True)
        test_demo_inputs = generate_detector_demo_inputs(train=False)
        losses = detector(**train_demo_inputs)
        assert isinstance(losses, dict)

        # Test forward test
        with torch.no_grad():
            _ = detector(**test_demo_inputs, return_loss=False)
