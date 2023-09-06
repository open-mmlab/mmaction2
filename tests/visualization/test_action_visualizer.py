# Copyright (c) OpenMMLab. All rights reserved.
import platform

import decord
import pytest

from mmaction.structures import ActionDataSample
from mmaction.visualization import ActionVisualizer


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_visualizer():
    video = decord.VideoReader('./demo/demo.mp4')
    video = video.get_batch(range(32)).asnumpy()

    data_sample = ActionDataSample()
    data_sample.set_gt_label(2)

    vis = ActionVisualizer()
    vis.add_datasample('demo', video)
    vis.add_datasample('demo', video, data_sample)
    vis.add_datasample('demo', video, data_sample, step=1)
    return
