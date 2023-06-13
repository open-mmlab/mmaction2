# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.models.localizers.utils import post_processing


def test_post_processing():
    # test with multiple results
    result = np.array([[0., 1., 1., 1., 0.5, 0.5], [0., 0.4, 1., 1., 0.4, 0.4],
                       [0., 0.95, 1., 1., 0.6, 0.6]])
    video_info = dict(
        video_name='v_test',
        duration_second=100,
        duration_frame=960,
        feature_frame=960)
    proposal_list = post_processing(result, video_info, 0.75, 0.65, 0.9, 2, 16)
    assert isinstance(proposal_list[0], dict)
    assert proposal_list[0]['score'] == 0.6
    assert proposal_list[0]['segment'] == [0., 95.0]
    assert isinstance(proposal_list[1], dict)
    assert proposal_list[1]['score'] == 0.4
    assert proposal_list[1]['segment'] == [0., 40.0]

    # test with only result
    result = np.array([[0., 1., 1., 1., 0.5, 0.5]])
    video_info = dict(
        video_name='v_test',
        duration_second=100,
        duration_frame=960,
        feature_frame=960)
    proposal_list = post_processing(result, video_info, 0.75, 0.65, 0.9, 1, 16)
    assert isinstance(proposal_list[0], dict)
    assert proposal_list[0]['score'] == 0.5
    assert proposal_list[0]['segment'] == [0., 100.0]
