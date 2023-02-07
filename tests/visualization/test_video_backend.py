# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import decord
import torch
from mmengine.structures import LabelData

from mmaction.structures import ActionDataSample
from mmaction.utils import register_all_modules
from mmaction.visualization import ActionVisualizer

register_all_modules()


def test_local_visbackend():
    video = decord.VideoReader('./demo/demo.mp4')
    video = video.get_batch(range(32)).asnumpy()

    data_sample = ActionDataSample()
    data_sample.gt_labels = LabelData(item=torch.tensor([2]))
    with TemporaryDirectory() as tmp_dir:
        vis = ActionVisualizer(
            save_dir=tmp_dir, vis_backends=[dict(type='LocalVisBackend')])
        vis.add_datasample('demo', video, data_sample)
        for k in range(32):
            frame_path = osp.join(tmp_dir, 'vis_data/demo/frames_0/%d.png' % k)
            assert Path(frame_path).exists()

        vis.add_datasample('demo', video, data_sample, step=1)
        for k in range(32):
            frame_path = osp.join(tmp_dir, 'vis_data/demo/frames_1/%d.png' % k)
            assert Path(frame_path).exists()
    return


def test_tensorboard_visbackend():
    video = decord.VideoReader('./demo/demo.mp4')
    video = video.get_batch(range(32)).asnumpy()

    data_sample = ActionDataSample()
    data_sample.gt_labels = LabelData(item=torch.tensor([2]))
    with TemporaryDirectory() as tmp_dir:
        vis = ActionVisualizer(
            save_dir=tmp_dir,
            vis_backends=[dict(type='TensorboardVisBackend')])
        vis.add_datasample('demo', video, data_sample, step=1)

        assert Path(osp.join(tmp_dir, 'vis_data')).exists()
        flag = False
        for item in os.listdir(osp.join(tmp_dir, 'vis_data')):
            if item.startswith('events.out.tfevents.'):
                flag = True
                break
        assert flag, 'Cannot find tensorboard file!'
        # wait tensorboard store asynchronously
        time.sleep(1)
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
