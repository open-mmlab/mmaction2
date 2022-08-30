# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmengine.dist import master_only
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
        video (Union[np.ndarray, Sequence[np.ndarray]]):
            the origin video to draw. The format should be RGB.
            For np.ndarray input, the video shape should be (N, H, W, C).
            For Sequence[np.ndarray] input, the shape of each frame in
             the sequence should be (H, W, C).
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
        >>> from mmaction.core import ActionDataSample, ActionVisualizer
        >>> from mmengine.structures import LabelData
        >>> # Example frame
        >>> video = decord.VideoReader('./demo/demo.mp4')
        >>> video = video.get_batch(range(32)).asnumpy()
        >>> # Example annotation
        >>> data_sample = ActionDataSample()
        >>> data_sample.gt_labels = LabelData(item=torch.tensor([2]))
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
        video: Optional[np.ndarray] = None,
        vis_backends: Optional[List[Dict]] = None,
        save_dir: Optional[str] = None,
        fig_save_cfg=dict(frameon=False),
        fig_show_cfg=dict(frameon=False, num='show')
    ) -> None:
        self._dataset_meta = None
        self._vis_backends = dict()

        if save_dir is None:
            warnings.warn('`Visualizer` backend is not initialized '
                          'because save_dir is None.')
        elif vis_backends is not None:
            assert len(vis_backends) > 0, 'empty list'
            names = [
                vis_backend.get('name', None) for vis_backend in vis_backends
            ]
            if None in names:
                if len(set(names)) > 1:
                    raise RuntimeError(
                        'If one of them has a name attribute, '
                        'all backends must use the name attribute')
                else:
                    type_names = [
                        vis_backend['type'] for vis_backend in vis_backends
                    ]
                    if len(set(type_names)) != len(type_names):
                        raise RuntimeError(
                            'The same vis backend cannot exist in '
                            '`vis_backend` config. '
                            'Please specify the name field.')

            if None not in names and len(set(names)) != len(names):
                raise RuntimeError('The name fields cannot be the same')

            save_dir = osp.join(save_dir, 'vis_data')

            for vis_backend in vis_backends:
                name = vis_backend.pop('name', vis_backend['type'])
                vis_backend.setdefault('save_dir', save_dir)
                self._vis_backends[name] = VISBACKENDS.build(vis_backend)

        self.is_inline = 'inline' in plt.get_backend()

        self.fig_save = None
        self.fig_show = None
        self.fig_save_num = fig_save_cfg.get('num', None)
        self.fig_show_num = fig_show_cfg.get('num', None)
        self.fig_save_cfg = fig_save_cfg
        self.fig_show_cfg = fig_show_cfg

        (self.fig_save_canvas, self.fig_save,
         self.ax_save) = self._initialize_fig(fig_save_cfg)
        self.dpi = self.fig_save.get_dpi()

    @master_only
    def add_datasample(self,
                       name: str,
                       video: Union[np.ndarray, Sequence[np.ndarray]],
                       data_sample: Optional[ActionDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_score: bool = True,
                       rescale_factor: Optional[float] = None,
                       show_frames: bool = False,
                       text_cfg: dict = dict(),
                       wait_time: float = 0.1,
                       out_folder: Optional[str] = None,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If ``out_folder`` is specified, all storage backends are ignored
          and save the videos to the ``out_folder``.
        - If ``show_frames`` is True, plot the frames in a window sequentially,
          please confirm you are able to access the graphical interface.

        Args:
            name (str): The frame identifier.
            video (np.ndarray): The video to draw.
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
            out_folder (str, optional): Extra folder to save the visualization
                result. If specified, the visualizer will only save the result
                frame to the out_folder and ignore its storage backends.
                Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = None
        wait_time_in_milliseconds = wait_time * 10**6
        tol_video = len(video)

        if self.dataset_meta is not None:
            classes = self.dataset_meta.get('classes', None)

        if data_sample is None:
            data_sample = ActionDataSample()

        resulted_video = []
        for frame_idx, frame in enumerate(video):
            frame_name = 'frame %d of %s' % (frame_idx, name)
            if rescale_factor is not None:
                frame = mmcv.imrescale(frame, rescale_factor)

            texts = ['Frame %d of total %d frames' % (frame_idx, tol_video)]
            self.set_image(frame)

            if draw_gt and 'gt_labels' in data_sample:
                gt_labels = data_sample.gt_labels
                idx = gt_labels.item.tolist()
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
            text_cfg = {
                'positions':
                np.array([(img_scale * 5, ) * 2]).astype(np.int32),
                'font_sizes': int(img_scale * 7),
                'font_families': 'monospace',
                'colors': 'white',
                'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
                **text_cfg
            }
            self.draw_texts('\n'.join(texts), **text_cfg)
            drawn_img = self.get_image()
            resulted_video.append(drawn_img)

            if show_frames:
                self.show(
                    drawn_img,
                    win_name=frame_name,
                    wait_time=wait_time_in_milliseconds)

        resulted_video = np.array(resulted_video)
        if out_folder is not None:
            resulted_video = resulted_video[..., ::-1]
            os.makedirs(out_folder, exist_ok=True)
            # save the frame to the target file instead of vis_backends
            for frame_idx, frame in enumerate(resulted_video):
                mmcv.imwrite(frame, out_folder + '/%d.png' % frame_idx)
        else:
            self.add_video(name, resulted_video, step=step)

    @master_only
    def add_video(self, name: str, image: np.ndarray, step: int = 0) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_video(name, image, step)  # type: ignore
