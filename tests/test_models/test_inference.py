import mmcv
import numpy as np
import pytest
import torch
import torch.nn as nn

from mmaction.apis import inference_recognizer, init_recognizer

video_config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'  # noqa: E501
frame_config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'  # noqa: E501
label_path = 'demo/label_map_k400.txt'
video_path = 'demo/demo.mp4'


def test_init_recognizer():
    with pytest.raises(TypeError):
        # config must be a filename or Config object
        init_recognizer(dict(config_file=None))

    with pytest.raises(RuntimeError):
        # input data type should be consist with the dataset type
        init_recognizer(frame_config_file)

    with pytest.raises(RuntimeError):
        # input data type should be consist with the dataset type
        init_recognizer(video_config_file, use_frames=True)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = init_recognizer(video_config_file, None, device)

    config = mmcv.Config.fromfile(video_config_file)
    config.model.backbone.pretrained = None

    isinstance(model, nn.Module)
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda is True
    else:
        assert next(model.parameters()).is_cuda is False
    assert model.cfg.model.backbone.pretrained is None


def test_inference_recognizer():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    model = init_recognizer(video_config_file, None, device)

    with pytest.raises(RuntimeError):
        # video path doesn't exist
        inference_recognizer(model, 'missing.mp4', label_path)

    with pytest.raises(RuntimeError):
        # ``video_path`` should be consist with the ``use_frames``
        inference_recognizer(model, video_path, label_path, use_frames=True)

    with pytest.raises(RuntimeError):
        # ``video_path`` should be consist with the ``use_frames``
        inference_recognizer(model, 'demo/', label_path)

    for ops in model.cfg.data.test.pipeline:
        if ops['type'] in ('TenCrop', 'ThreeCrop'):
            # Use CenterCrop to reduce memory in order to pass CI
            ops['type'] = 'CenterCrop'

    top5_label = inference_recognizer(model, video_path, label_path)
    scores = [item[1] for item in top5_label]
    assert len(top5_label) == 5
    assert scores == sorted(scores, reverse=True)

    _, feat = inference_recognizer(
        model,
        video_path,
        label_path,
        outputs=('backbone', 'cls_head'),
        as_tensor=False)
    assert isinstance(feat, dict)
    assert 'backbone' in feat and 'cls_head' in feat
    assert isinstance(feat['backbone'], np.ndarray)
    assert isinstance(feat['cls_head'], np.ndarray)
    assert feat['backbone'].shape == (25, 2048, 7, 7)
    assert feat['cls_head'].shape == (1, 400)

    _, feat = inference_recognizer(
        model,
        video_path,
        label_path,
        outputs=('backbone.layer3', 'backbone.layer3.1.conv1'))
    assert 'backbone.layer3.1.conv1' in feat and 'backbone.layer3' in feat
    assert isinstance(feat['backbone.layer3.1.conv1'], torch.Tensor)
    assert isinstance(feat['backbone.layer3'], torch.Tensor)
    assert feat['backbone.layer3'].size() == (25, 1024, 14, 14)
    assert feat['backbone.layer3.1.conv1'].size() == (25, 256, 14, 14)

    cfg_file = 'configs/recognition/slowfast/slowfast_r50_video_inference_4x16x1_256e_kinetics400_rgb.py'  # noqa: E501
    sf_model = init_recognizer(cfg_file, None, device)
    for ops in sf_model.cfg.data.test.pipeline:
        # Changes to reduce memory in order to pass CI
        if ops['type'] in ('TenCrop', 'ThreeCrop'):
            ops['type'] = 'CenterCrop'
        if ops['type'] == 'SampleFrames':
            ops['num_clips'] = 1
    _, feat = inference_recognizer(
        sf_model, video_path, label_path, outputs=('backbone', 'cls_head'))
    assert isinstance(feat, dict) and isinstance(feat['backbone'], tuple)
    assert 'backbone' in feat and 'cls_head' in feat
    assert len(feat['backbone']) == 2
    assert isinstance(feat['backbone'][0], torch.Tensor)
    assert isinstance(feat['backbone'][1], torch.Tensor)
    assert feat['backbone'][0].size() == (1, 2048, 4, 8, 8)
    assert feat['backbone'][1].size() == (1, 256, 32, 8, 8)
    assert feat['cls_head'].size() == (1, 400)
