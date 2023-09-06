# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.fileio.io import isdir, isfile, join_path, list_dir_or_file
from mmengine.visualization import Visualizer

from mmaction.registry import VISBACKENDS, VISUALIZERS
from mmaction.structures import ActionDataSample


def _get_adaptive_scale(img_shape: Tuple[int, int],
                        min_scale: float = 0.3,
                        max_scale: float = 3.0) -> float:
    """Get adaptive scale according to frame shape.

    The target scale depends on the the short edge length of the frame. If the
    short edge length equals 224, the output is 1.0. And output linear scales
    according the short edge length.

    You can also specify the minimum scale and the maximum scale to limit the
    linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas frame.
        min_size (int): The minimum scale. Defaults to 0.3.
        max_size (int): The maximum scale. Defaults to 3.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


@VISUALIZERS.register_module()
class ActionVisualizer(Visualizer):
    """Universal Visualizer for classification task.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.

    Examples:
        >>> import torch
        >>> import decord
        >>> from pathlib import Path
        >>> from mmaction.structures import ActionDataSample, ActionVisualizer
        >>> from mmengine.structures import LabelData
        >>> # Example frame
        >>> video = decord.VideoReader('./demo/demo.mp4')
        >>> video = video.get_batch(range(32)).asnumpy()
        >>> # Example annotation
        >>> data_sample = ActionDataSample()
        >>> data_sample.gt_label = LabelData(item=torch.tensor([2]))
        >>> # Setup the visualizer
        >>> vis = ActionVisualizer(
        ...     save_dir="./outputs",
        ...     vis_backends=[dict(type='LocalVisBackend')])
        >>> # Set classes names
        >>> vis.dataset_meta = {'classes': ['running', 'standing', 'sitting']}
        >>> # Save the visualization result by the specified storage backends.
        >>> vis.add_datasample('demo', video)
        >>> assert Path('outputs/vis_data/demo/frames_0/1.png').exists()
        >>> assert Path('outputs/vis_data/demo/frames_0/2.png').exists()
        >>> # Save another visualization result with the same name.
        >>> vis.add_datasample('demo', video, step=1)
        >>> assert Path('outputs/vis_data/demo/frames_1/2.png').exists()
    """

    def __init__(
        self,
        name='visualizer',
        vis_backends: Optional[List[Dict]] = None,
        save_dir: Optional[str] = None,
        fig_save_cfg=dict(frameon=False),
        fig_show_cfg=dict(frameon=False)
    ) -> None:
        super().__init__(
            name=name,
            image=None,
            vis_backends=vis_backends,
            save_dir=save_dir,
            fig_save_cfg=fig_save_cfg,
            fig_show_cfg=fig_show_cfg)

    def _load_video(self,
                    video: Union[np.ndarray, Sequence[np.ndarray], str],
                    target_resolution: Optional[Tuple[int]] = None):
        """Load video from multiple source and convert to target resolution.

        Args:
            video (np.ndarray, str): The video to draw.
            target_resolution (Tuple[int], optional): Set to
                (desired_width desired_height) to have resized frames. If
                either dimension is None, the frames are resized by keeping
                the existing aspect ratio. Defaults to None.
        """
        if isinstance(video, np.ndarray) or isinstance(video, list):
            frames = video
        elif isinstance(video, str):
            # video file path
            if isfile(video):
                try:
                    import decord
                except ImportError:
                    raise ImportError(
                        'Please install decord to load video file.')
                video = decord.VideoReader(video)
                frames = [x.asnumpy()[..., ::-1] for x in video]
            # rawframes folder path
            elif isdir(video):
                frame_list = sorted(list_dir_or_file(video, list_dir=False))
                frames = [mmcv.imread(join_path(video, x)) for x in frame_list]
        else:
            raise TypeError(f'type of video {type(video)} not supported')

        if target_resolution is not None:
            w, h = target_resolution
            frame_h, frame_w, _ = frames[0].shape
            if w == -1:
                w = int(h / frame_h * frame_w)
            if h == -1:
                h = int(w / frame_w * frame_h)
            frames = [mmcv.imresize(f, (w, h)) for f in frames]

        return frames

    @master_only
    def add_datasample(self,
                       name: str,
                       video: Union[np.ndarray, Sequence[np.ndarray], str],
                       data_sample: Optional[ActionDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_score: bool = True,
                       rescale_factor: Optional[float] = None,
                       show_frames: bool = False,
                       text_cfg: dict = dict(),
                       wait_time: float = 0.1,
                       out_path: Optional[str] = None,
                       out_type: str = 'img',
                       target_resolution: Optional[Tuple[int]] = None,
                       step: int = 0,
                       fps: int = 4) -> None:
        """Draw datasample and save to all backends.

        - If ``out_path`` is specified, all storage backends are ignored
          and save the videos to the ``out_path``.
        - If ``show_frames`` is True, plot the frames in a window sequentially,
          please confirm you are able to access the graphical interface.

        Args:
            name (str): The frame identifier.
            video (np.ndarray, str): The video to draw. supports decoded
                np.ndarray, video file path, rawframes folder path.
            data_sample (:obj:`ActionDataSample`, optional): The annotation of
                the frame. Defaults to None.
            draw_gt (bool): Whether to draw ground truth labels.
                Defaults to True.
            draw_pred (bool): Whether to draw prediction labels.
                Defaults to True.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            rescale_factor (float, optional): Rescale the frame by the rescale
                factor before visualization. Defaults to None.
            show_frames (bool): Whether to display the frames of the video.
                Defaults to False.
            text_cfg (dict): Extra text setting, which accepts
                arguments of :attr:`mmengine.Visualizer.draw_texts`.
                Defaults to an empty dict.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.1.
            out_path (str, optional): Extra folder to save the visualization
                result. If specified, the visualizer will only save the result
                frame to the out_path and ignore its storage backends.
                Defaults to None.
            out_type (str): Output format type, choose from 'img', 'gif',
                'video'. Defaults to ``'img'``.
            target_resolution (Tuple[int], optional): Set to
                (desired_width desired_height) to have resized frames. If
                either dimension is None, the frames are resized by keeping
                the existing aspect ratio. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
            fps (int): Frames per second for saving video. Defaults to 4.
        """
        classes = None
        video = self._load_video(video, target_resolution)
        tol_video = len(video)

        if self.dataset_meta is not None:
            classes = self.dataset_meta.get('classes', None)

        if data_sample is None:
            data_sample = ActionDataSample()

        resulted_video = []
        for frame_idx, frame in enumerate(video):
            frame_name = 'frame %d of %s' % (frame_idx + 1, name)
            if rescale_factor is not None:
                frame = mmcv.imrescale(frame, rescale_factor)

            texts = ['Frame %d of total %d frames' % (frame_idx, tol_video)]
            self.set_image(frame)

            if draw_gt and 'gt_labels' in data_sample:
                gt_labels = data_sample.gt_label
                idx = gt_labels.tolist()
                class_labels = [''] * len(idx)
                if classes is not None:
                    class_labels = [f' ({classes[i]})' for i in idx]
                labels = [
                    str(idx[i]) + class_labels[i] for i in range(len(idx))
                ]
                prefix = 'Ground truth: '
                texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

            if draw_pred and 'pred_labels' in data_sample:
                pred_labels = data_sample.pred_labels
                idx = pred_labels.item.tolist()
                score_labels = [''] * len(idx)
                class_labels = [''] * len(idx)
                if draw_score and 'score' in pred_labels:
                    score_labels = [
                        f', {pred_labels.score[i].item():.2f}' for i in idx
                    ]

                if classes is not None:
                    class_labels = [f' ({classes[i]})' for i in idx]

                labels = [
                    str(idx[i]) + score_labels[i] + class_labels[i]
                    for i in range(len(idx))
                ]
                prefix = 'Prediction: '
                texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

            img_scale = _get_adaptive_scale(frame.shape[:2])
            _text_cfg = {
                'positions':
                np.array([(img_scale * 5, ) * 2]).astype(np.int32),
                'font_sizes': int(img_scale * 7),
                'font_families': 'monospace',
                'colors': 'white',
                'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
            }
            _text_cfg.update(text_cfg)
            self.draw_texts('\n'.join(texts), **_text_cfg)
            drawn_img = self.get_image()
            resulted_video.append(drawn_img)

        if show_frames:
            frame_wait_time = 1. / fps
            for frame_idx, drawn_img in enumerate(resulted_video):
                frame_name = 'frame %d of %s' % (frame_idx + 1, name)
                if frame_idx < len(resulted_video) - 1:
                    wait_time = frame_wait_time
                else:
                    wait_time = wait_time
                self.show(
                    drawn_img[:, :, ::-1],
                    win_name=frame_name,
                    wait_time=wait_time)

        resulted_video = np.array(resulted_video)
        if out_path is not None:
            save_dir, save_name = osp.split(out_path)
            vis_backend_cfg = dict(type='LocalVisBackend', save_dir=save_dir)
            tmp_local_vis_backend = VISBACKENDS.build(vis_backend_cfg)
            tmp_local_vis_backend.add_video(
                save_name,
                resulted_video,
                step=step,
                fps=fps,
                out_type=out_type)
        else:
            self.add_video(
                name, resulted_video, step=step, fps=fps, out_type=out_type)
        return resulted_video

    @master_only
    def add_video(
        self,
        name: str,
        image: np.ndarray,
        step: int = 0,
        fps: int = 4,
        out_type: str = 'img',
    ) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
            fps (int): Frames per second for saving video. Defaults to 4.
            out_type (str): Output format type, choose from 'img', 'gif',
                'video'. Defaults to ``'img'``.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_video(
                name, image, step=step, fps=fps,
                out_type=out_type)  # type: ignore
