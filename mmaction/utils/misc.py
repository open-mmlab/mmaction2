# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import inspect
import os
import os.path as osp
import random
import string
from types import FunctionType, ModuleType
from typing import Optional, Union

import cv2
import mmcv
import numpy as np


def get_random_string(length: int = 15) -> str:
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Defaults to 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id() -> int:
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir() -> str:
    """Get shm dir for temporary usage."""
    return '/dev/shm'


def frame_extract(video_path: str,
                  short_side: Optional[int] = None,
                  out_dir: str = './tmp'):
    """Extract frames given video_path.

    Args:
        video_path (str): The video path.
        short_side (int): Target short-side of the output image.
            Defaults to None, means keeping original shape.
        out_dir (str): The output directory. Defaults to ``'./tmp'``.
    """
    # Load the video, extract frames into OUT_DIR/video_name
    target_dir = osp.join(out_dir, osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    assert osp.exists(video_path), f'file not exit {video_path}'
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if short_side is not None:
            if new_h is None:
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


class VideoWriter():

    def __init__(self, video_file, fps):
        self.video_file = video_file
        self.fps = fps
        if video_file.endswith('.mp4'):
            self.fourcc = 'mp4v'
        elif video_file.endswith('.avi'):
            self.fourcc = 'XVID'

        out_dir = osp.dirname(osp.abspath(self.video_file))
        if not osp.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    def _init_cv2_writer(self, frame):
        from cv2 import VideoWriter, VideoWriter_fourcc
        height, width = frame.shape[:2]
        resolution = (width, height)
        self.writer = VideoWriter(self.video_file,
                                  VideoWriter_fourcc(*self.fourcc), self.fps,
                                  resolution)

    def write_frame(self, frame):
        if not getattr(self, 'writer', None):
            self._init_cv2_writer(frame)
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self.release()


def get_str_type(module: Union[str, ModuleType, FunctionType]) -> str:
    """Return the string type name of module.

    Args:
        module (str | ModuleType | FunctionType):
            The target module class

    Returns:
        Class name of the module
    """
    if isinstance(module, str):
        str_type = module
    elif inspect.isclass(module) or inspect.isfunction(module):
        str_type = module.__name__
    else:
        return None

    return str_type
