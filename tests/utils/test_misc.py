# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from tempfile import TemporaryDirectory

from mmaction.utils import frame_extract


def test_frame_extract():
    data_prefix = osp.normpath(osp.join(osp.dirname(__file__), '../data'))
    video_path = osp.join(data_prefix, 'test.mp4')
    with TemporaryDirectory() as tmp_dir:
        # assign short_side
        frame_paths, frames = frame_extract(
            video_path, short_side=100, out_dir=tmp_dir)
        assert osp.exists(tmp_dir) and \
            len(os.listdir(f'{tmp_dir}/test')) == len(frame_paths)
        assert min(frames[0].shape[:2]) == 100
        # default short_side
        frame_paths, frames = frame_extract(video_path, out_dir=tmp_dir)
        assert osp.exists(tmp_dir) and \
            len(os.listdir(f'{tmp_dir}/test')) == len(frame_paths)
