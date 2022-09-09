# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.utils import register_all_modules

video_config_file = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'  # noqa: E501
video_path = 'demo/demo.mp4'

register_all_modules()


def test_init_recognizer():
    with pytest.raises(TypeError):
        # config must be a filename or Config object
        init_recognizer(dict(config_file=None))

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = init_recognizer(video_config_file, None, device)

    isinstance(model, nn.Module)
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda is True
    else:
        assert next(model.parameters()).is_cuda is False
    assert model.cfg.model.backbone.pretrained is None


def test_video_inference_recognizer():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    model = init_recognizer(video_config_file, None, device)

    with pytest.raises(FileNotFoundError):
        # video path doesn't exist
        inference_recognizer(model, 'missing.mp4')

    for ops in model.cfg.test_pipeline:
        if ops['type'] in ('TenCrop', 'ThreeCrop'):
            # Use CenterCrop to reduce memory in order to pass CI
            ops['type'] = 'CenterCrop'

    top5_label = inference_recognizer(model, video_path)
    scores = [item[1] for item in top5_label]
    assert len(top5_label) == 5
    assert scores == sorted(scores, reverse=True)
