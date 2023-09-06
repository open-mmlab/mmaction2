# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
from mmaction.testing import get_skeletongcn_cfg
from mmaction.utils import register_all_modules


def train_test_step(cfg, input_shape):
    recognizer = MODELS.build(cfg.model)
    num_classes = cfg.model.cls_head.num_classes
    data_batch = {
        'inputs': [torch.randn(input_shape)],
        'data_samples': [ActionDataSample().set_gt_label(2)]
    }

    # test train_step
    optim_wrapper = MagicMock()
    loss_vars = recognizer.train_step(data_batch, optim_wrapper)
    assert 'loss' in loss_vars
    assert 'loss_cls' in loss_vars
    optim_wrapper.update_params.assert_called_once()

    # test test_step
    with torch.no_grad():
        predictions = recognizer.test_step(data_batch)
    score = predictions[0].pred_score
    assert len(predictions) == 1
    assert score.shape == torch.Size([num_classes])
    assert torch.min(score) >= 0
    assert torch.max(score) <= 1

    # test when average_clips is None
    recognizer.cls_head.average_clips = None
    num_clips = 3
    input_shape = (num_clips, *input_shape[1:])
    data_batch['inputs'] = [torch.randn(input_shape)]
    with torch.no_grad():
        predictions = recognizer.test_step(data_batch)
    score = predictions[0].pred_score
    assert len(predictions) == 1
    assert score.shape == torch.Size([num_clips, num_classes])

    return loss_vars, predictions


def test_stgcn():
    register_all_modules()
    config = get_skeletongcn_cfg(
        'stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py')
    input_shape = (1, 2, 30, 17, 3)  # N M T V C
    train_test_step(config, input_shape=input_shape)


def test_agcn():
    register_all_modules()
    config = get_skeletongcn_cfg(
        '2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py')
    input_shape = (1, 2, 30, 17, 3)  # N M T V C
    train_test_step(config, input_shape=input_shape)


def test_stgcn_plusplus():
    register_all_modules()
    config = get_skeletongcn_cfg(
        'stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py')
    input_shape = (1, 2, 30, 17, 3)  # N M T V C
    train_test_step(config, input_shape=input_shape)
