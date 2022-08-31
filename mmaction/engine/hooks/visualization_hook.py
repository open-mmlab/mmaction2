# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
from typing import Optional, Sequence

from mmengine import FileClient
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from mmaction.registry import HOOKS
from mmaction.structures import ActionDataSample


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Classification Visualization Hook. Used to visualize validation and
    testing prediction results.

    - If ``out_dir`` is specified, all storage backends are ignored
      and save the image to the ``out_dir``.
    - If ``show`` is True, plot the result image in a window, please
      confirm you are able to access the graphical interface.
    Args:
        enable (bool): Whether to enable this hook. Defaults to False.
        interval (int): The interval of samples to visualize. Defaults to 5000.
        show (bool): Whether to display the drawn image. Defaults to False.
        out_dir (str, optional): directory where painted images will be saved
            in the testing process. If None, handle with the backends of the
            visualizer. Defaults to None.
        **kwargs: other keyword arguments of
            :meth:`mmcls.visualization.ClsVisualizer.add_datasample`.
    """

    def __init__(self,
                 enable=False,
                 interval: int = 5000,
                 show: bool = False,
                 out_dir: Optional[str] = None,
                 **kwargs):
        self._visualizer: Visualizer = Visualizer.get_current_instance()

        self.enable = enable
        self.interval = interval
        self.show = show
        self.out_dir = out_dir
        if out_dir is not None:
            self.file_client = FileClient.infer_client(uri=out_dir)
        else:
            self.file_client = None

        self.draw_args = {**kwargs, 'show': show}

    def _draw_samples(self,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[ActionDataSample],
                      step: int = 0) -> None:
        """Visualize every ``self.interval`` samples from a data batch.

        Args:
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ActionDataSample`]): Outputs from model.
            step (int): Global step value to record. Defaults to 0.
        """
        if self.enable is False:
            return

        batch_size = len(data_samples)
        videos = data_batch['inputs']
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            video = videos[sample_id - start_idx]
            # move channel to the last
            video = video.permute(1, 2, 3, 0).numpy().astype('uint8')

            data_sample = data_samples[sample_id - start_idx]
            if 'filename' in data_sample:
                # osp.basename works on different platforms even file clients.
                sample_name = osp.basename(data_sample.get('filename'))
            elif 'frame_dir' in data_sample:
                sample_name = osp.basename(data_sample.get('frame_dir'))
            else:
                sample_name = str(sample_id)

            draw_args = self.draw_args
            if self.out_dir is not None:
                draw_args['out_folder'] = self.file_client.join_path(
                    self.out_dir, f'{sample_name}_{step}')

            self._visualizer.add_datasample(
                sample_name,
                video=video,
                data_sample=data_sample,
                step=step,
                **self.draw_args,
            )

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[ActionDataSample]) -> None:
        """Visualize every ``self.interval`` samples during validation.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`ActionDataSample`]): Outputs from model.
        """
        if isinstance(runner.train_loop, EpochBasedTrainLoop):
            step = runner.epoch
        else:
            step = runner.iter

        self._draw_samples(batch_idx, data_batch, outputs, step=step)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[ActionDataSample]) -> None:
        """Visualize every ``self.interval`` samples during test.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): Outputs from model.
        """
        self._draw_samples(batch_idx, data_batch, outputs, step=0)
