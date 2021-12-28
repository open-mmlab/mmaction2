# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.models import build_recognizer
from ..base import generate_recognizer_demo_inputs, get_skeletongcn_cfg


def test_skeletongcn():
    config = get_skeletongcn_cfg('stgcn/stgcn_80e_ntu60_xsub_keypoint.py')
    with pytest.raises(TypeError):
        # "pretrained" must be a str or None
        config.model['backbone']['pretrained'] = ['None']
        recognizer = build_recognizer(config.model)

    config.model['backbone']['pretrained'] = None
    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 300, 17, 2)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, 'skeleton')

    skeletons = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(skeletons, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        skeleton_list = [skeleton[None, :] for skeleton in skeletons]
        for one_skeleton in skeleton_list:
            recognizer(one_skeleton, None, return_loss=False)

    # test stgcn without edge importance weighting
    config.model['backbone']['edge_importance_weighting'] = False
    recognizer = build_recognizer(config.model)

    input_shape = (1, 3, 300, 17, 2)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, 'skeleton')

    skeletons = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(skeletons, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        skeleton_list = [skeleton[None, :] for skeleton in skeletons]
        for one_skeleton in skeleton_list:
            recognizer(one_skeleton, None, return_loss=False)
