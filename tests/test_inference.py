import mmcv
import pytest
import torch
import torch.nn as nn

from mmaction.apis import inference_recognizer, init_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'  # noqa: E501
label_path = 'demo/label_map.txt'
video_path = 'demo/demo.mp4'


def test_init_recognizer():
    with pytest.raises(TypeError):
        init_recognizer(dict(config_file=None))

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = init_recognizer(config_file, None, device)

    config = mmcv.Config.fromfile(config_file)
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
    model = init_recognizer(config_file, None, device)

    for ops in model.cfg.data.test.pipeline:
        if ops['type'] == 'TenCrop':
            # Use CenterCrop to reduce memory in order to pass CI
            ops['type'] = 'CenterCrop'

    top5_label = inference_recognizer(model, video_path, label_path)
    scores = [item[1] for item in top5_label]
    assert len(top5_label) == 5
    assert scores == sorted(scores, reverse=True)
