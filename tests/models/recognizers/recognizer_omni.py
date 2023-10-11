# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_recognizer_cfg
from mmaction.utils import register_all_modules


def test_omni_resnet():
    register_all_modules()
    config = get_recognizer_cfg(
        'omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py')
    recognizer = MODELS.build(config.model)

    # test train_step

    video_sample = {
        'inputs': [
            torch.randint(0, 255, (1, 3, 8, 224, 224)),
            torch.randint(0, 255, (1, 3, 8, 224, 224))
        ],
        'data_samples': [
            ActionDataSample().set_gt_label(2),
            ActionDataSample().set_gt_label(2)
        ]
    }

    image_sample = {
        'inputs': [
            torch.randint(0, 255, (1, 3, 224, 224)),
            torch.randint(0, 255, (1, 3, 224, 224))
        ],
        'data_samples': [
            ActionDataSample().set_gt_label(2),
            ActionDataSample().set_gt_label(2)
        ]
    }

    optim_wrapper = MagicMock()
    loss_vars = recognizer.train_step([video_sample, image_sample],
                                      optim_wrapper)
    assert 'loss_cls_0' in loss_vars
    assert 'loss_cls_1' in loss_vars

    loss_vars = recognizer.train_step([image_sample, video_sample],
                                      optim_wrapper)
    assert 'loss_cls_0' in loss_vars
    assert 'loss_cls_1' in loss_vars

    # test test_step
    with torch.no_grad():
        predictions = recognizer.test_step(video_sample)
    score = predictions[0].pred_score
    assert len(predictions) == 2
    assert torch.min(score) >= 0
    assert torch.max(score) <= 1
