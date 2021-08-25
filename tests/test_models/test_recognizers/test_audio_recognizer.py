# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import build_recognizer
from ..base import generate_recognizer_demo_inputs, get_audio_recognizer_cfg


def test_audio_recognizer():
    config = get_audio_recognizer_cfg(
        'resnet/tsn_r18_64x1x1_100e_kinetics400_audio_feature.py')
    config.model['backbone']['pretrained'] = None

    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 1, 128, 80)
    demo_inputs = generate_recognizer_demo_inputs(
        input_shape, model_type='audio')

    audios = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(audios, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        audio_list = [audio[None, :] for audio in audios]
        for one_spectro in audio_list:
            recognizer(one_spectro, None, return_loss=False)
