# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

import decord
import torch
from mmengine.structures import LabelData

from mmaction.structures import ActionDataSample
from mmaction.visualization import ActionVisualizer


def test_local_visbackend():
    video = decord.VideoReader('./demo/demo.mp4')
    video = video.get_batch(range(32)).asnumpy()

    data_sample = ActionDataSample()
    data_sample.gt_labels = LabelData(item=torch.tensor([2]))

    vis = ActionVisualizer(
        save_dir='./outputs', vis_backends=[dict(type='LocalVisBackend')])
    vis.add_datasample('demo', video, data_sample)
    for k in range(32):
        frame_path = 'outputs/vis_data/demo/frames_0/%d.png' % k
        assert Path(frame_path).exists()

    vis.add_datasample('demo', video, data_sample, step=1)
    for k in range(32):
        frame_path = 'outputs/vis_data/demo/frames_1/%d.png' % k
        assert Path(frame_path).exists()
    return


def test_tensorboard_visbackend():
    video = decord.VideoReader('./demo/demo.mp4')
    video = video.get_batch(range(32)).asnumpy()

    data_sample = ActionDataSample()
    data_sample.gt_labels = LabelData(item=torch.tensor([2]))

    vis = ActionVisualizer(
        save_dir='./outputs',
        vis_backends=[dict(type='TensorboardVisBackend')])
    vis.add_datasample('demo', video, data_sample, step=1)

    assert Path('outputs/vis_data/').exists()
    flag = False
    for item in os.listdir('outputs/vis_data/'):
        if item.startswith('events.out.tfevents.'):
            flag = True
            break
    assert flag, 'Cannot find tensorboard file!'
    return


"""
def test_wandb_visbackend():
    video = decord.VideoReader('./demo/demo.mp4')
    video = video.get_batch(range(32)).asnumpy()

    data_sample = ActionDataSample()
    data_sample.gt_labels = LabelData(item=torch.tensor([2]))

    vis = ActionVisualizer(
        save_dir='./outputs', vis_backends=[dict(type='WandbVisBackend')])
    vis.add_datasample('demo', video, data_sample, step=1)

    wandb_dir = 'outputs/vis_data/wandb/'
    assert Path(wandb_dir).exists()

    flag = False
    for item in os.listdir(wandb_dir):
        if item.startswith('run-') and os.path.isdir('%s/%s' %
                                                     (wandb_dir, item)):
            flag = True
            break
    assert flag, 'Cannot find wandb folder!'
    return
"""
