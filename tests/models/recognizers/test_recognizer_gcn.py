# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from ..base import generate_recognizer_demo_inputs, get_skeletongcn_cfg


def test_recognizer_gcn():
    register_all_modules()
    config = get_skeletongcn_cfg(
        'stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py')
    """
    TODO
    with pytest.raises(TypeError):
        # "pretrained" must be a str or None
        config.model['backbone']['pretrained'] = ['None']
        recognizer = MODELS.build(config.model)
    """

    config.model['backbone']['pretrained'] = None
    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 300, 17, 2)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, 'skeleton')

    skeletons = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(skeletons, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        skeleton_list = [skeleton[None, :] for skeleton in skeletons]
        for one_skeleton in skeleton_list:
            recognizer(one_skeleton, None, return_loss=False)

    # test stgcn without edge importance weighting
    config.model['backbone']['edge_importance_weighting'] = False
    recognizer = MODELS.build(config.model)

    input_shape = (1, 3, 300, 17, 2)
    demo_inputs = generate_recognizer_demo_inputs(input_shape, 'skeleton')

    skeletons = demo_inputs['imgs']
    gt_labels = demo_inputs['gt_labels']

    losses = recognizer(skeletons, gt_labels)
    assert isinstance(losses, torch.Tensor)

    # Test forward test
    with torch.no_grad():
        skeleton_list = [skeleton[None, :] for skeleton in skeletons]
        for one_skeleton in skeleton_list:
            recognizer(one_skeleton, None, return_loss=False)
