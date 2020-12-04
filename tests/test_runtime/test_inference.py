import mmcv
import pytest
import torch
import torch.nn as nn

from mmaction.apis import inference_recognizer, init_recognizer

video_config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'  # noqa: E501
frame_config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'  # noqa: E501
label_path = 'demo/label_map.txt'
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
