# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional

import cv2
import numpy as np
from mmengine.visualization import (LocalVisBackend, TensorboardVisBackend,
                                    WandbVisBackend)
from mmengine.visualization.vis_backend import force_init_env

from mmaction.registry import VISBACKENDS

try:
    import wandb
except ImportError:
    pass


@VISBACKENDS.register_module()
class LocalVisBackend(LocalVisBackend):
    """Local visualization backend class with video support.

    See mmengine.visualization.LocalVisBackend for more details.
    """

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  step: int = 0,
                  fps: Optional[int] = 4,
                  out_type: Optional[int] = 'img',
                  **kwargs) -> None:
        """Record the frames of a video to disk.

        Args:
            name (str): The video identifier (frame folder).
            frames (np.ndarray): The frames to be saved. The format
                should be RGB. The shape should be (T, H, W, C).
            step (int): Global step value to record. Defaults to 0.
            out_type (str): Output format type, choose from 'img', 'gif',
            'video'. Defaults to ``'img'``.
            fps (int): Frames per second for saving video. Defaults to 4.
        """
        assert frames.dtype == np.uint8

        if out_type == 'img':
            frames_dir = osp.join(self._save_dir, name, f'frames_{step}')
            os.makedirs(frames_dir, exist_ok=True)
            for idx, frame in enumerate(frames):
                drawn_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                save_file_name = f'{idx}.png'
                cv2.imwrite(osp.join(frames_dir, save_file_name), drawn_image)
        else:
            try:
                from moviepy.editor import ImageSequenceClip
            except ImportError:
                raise ImportError('Please install moviepy to enable '
                                  'output file.')

            frames = [x[..., ::-1] for x in frames]
            video_clips = ImageSequenceClip(frames, fps=fps)
            name = osp.splitext(name)[0]
            if out_type == 'gif':
                out_path = osp.join(self._save_dir, name + '.gif')
                video_clips.write_gif(out_path, logger=None)
            elif out_type == 'video':
                out_path = osp.join(self._save_dir, name + '.mp4')
                video_clips.write_videofile(
                    out_path, remove_temp=True, logger=None)


@VISBACKENDS.register_module()
class WandbVisBackend(WandbVisBackend):
    """Wandb visualization backend class with video support. See
    mmengine.visualization.WandbVisBackend for more details.

    Note that this requires the ``wandb`` and ``moviepy`` package. A wandb
    account login is also required at ``https://wandb.ai/authorize``.
    """

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  fps: int = 4,
                  **kwargs) -> None:
        """Record the frames of a video to wandb.

        Note that this requires the ``moviepy`` package.

        Args:
            name (str): The video identifier (frame folder).
            frames (np.ndarray): The frames to be saved. The format
                should be RGB. The shape should be (T, H, W, C).
            step is a useless parameter that Wandb does not need.
            fps (int): Frames per second. Defaults to 4.
        """
        frames = frames.transpose(0, 3, 1, 2)
        self._wandb.log({'video': wandb.Video(frames, fps=fps, format='gif')})


@VISBACKENDS.register_module()
class TensorboardVisBackend(TensorboardVisBackend):
    """Tensorboard visualization backend class with video support. See
    mmengine.visualization.TensorboardVisBackend for more details.

    Note that this requires the ``future`` and ``tensorboard`` package.
    """

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  step: int = 0,
                  fps: int = 4,
                  **kwargs) -> None:
        """Record the frames of a video to tensorboard.

        Note that this requires the ``moviepy`` package.

        Args:
            name (str): The video identifier (frame folder).
            frames (np.ndarray): The frames to be saved. The format
                should be RGB. The shape should be (T, H, W, C).
            step (int): Global step value to record. Defaults to 0.
            fps (int): Frames per second. Defaults to 4.
        """
        frames = frames.transpose(0, 3, 1, 2)
        frames = frames.reshape(1, *frames.shape)
        self._tensorboard.add_video(name, frames, global_step=step, fps=fps)
