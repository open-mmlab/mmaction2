# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

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

    @force_init_env
    def add_video(self,
                  name: str,
                  frames: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the frames of a video to disk.

        Args:
            name (str): The video identifier (frame folder).
            frames (np.ndarray): The frames to be saved. The format
                should be RGB. The shape should be (T, H, W, C).
            step (int): Global step value to record. Defaults to 0.
        """
        assert frames.dtype == np.uint8

        frames_dir = osp.join(self._save_dir, name, f'frames_{step}')
        os.makedirs(frames_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            drawn_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            save_file_name = f'{idx}.png'
            cv2.imwrite(osp.join(frames_dir, save_file_name), drawn_image)


@VISBACKENDS.register_module()
class WandbVisBackend(WandbVisBackend):

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
        """
        frames = frames.transpose(0, 3, 1, 2)
        self._wandb.log({'video': wandb.Video(frames, fps=fps, format='gif')})


@VISBACKENDS.register_module()
class TensorboardVisBackend(TensorboardVisBackend):

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
