# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import Hook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook, StepLrUpdaterHook
from torch.nn.modules.utils import _ntuple

from mmaction.core.lr import RelativeStepLrUpdaterHook
from mmaction.utils import get_root_logger


def modify_subbn3d_num_splits(logger, module, num_splits):
    """Recursively modify the number of splits of subbn3ds in module.
    Inheritates the running_mean and running_var from last subbn.bn.

    Args:
        logger (:obj:`logging.Logger`): The logger to log information.
        module (nn.Module): The module to be modified.
        num_splits (int): The targeted number of splits.
    Returns:
        int: The number of subbn3d modules modified.
    """
    count = 0
    for child in module.children():
        from mmaction.models import SubBatchNorm3D
        if isinstance(child, SubBatchNorm3D):
            new_split_bn = nn.BatchNorm3d(
                child.num_features * num_splits, affine=False).cuda()
            new_state_dict = new_split_bn.state_dict()

            for param_name, param in child.bn.state_dict().items():
                origin_param_shape = param.size()
                new_param_shape = new_state_dict[param_name].size()
                if len(origin_param_shape) == 1 and len(
                        new_param_shape
                ) == 1 and new_param_shape[0] >= origin_param_shape[
                        0] and new_param_shape[0] % origin_param_shape[0] == 0:
                    # weight bias running_var running_mean
                    new_state_dict[param_name] = torch.cat(
                        [param] *
                        (new_param_shape[0] // origin_param_shape[0]))
                else:
                    logger.info(f'skip  {param_name}')

            child.num_splits = num_splits
            new_split_bn.load_state_dict(new_state_dict)
            child.split_bn = new_split_bn
            count += 1
        else:
            count += modify_subbn3d_num_splits(logger, child, num_splits)
    return count


class LongShortCycleHook(Hook):
    """A multigrid method for efficiently training video models.

    This hook defines multigrid training schedule and update cfg
        accordingly, which is proposed in `A Multigrid Method for Efficiently
        Training Video Models <https://arxiv.org/abs/1912.00998>`_.

    Args:
        cfg (:obj:`mmcv.ConfigDictg`): The whole config for the experiment.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.multi_grid_cfg = cfg.get('multigrid', None)
        self.data_cfg = cfg.get('data', None)
        assert (self.multi_grid_cfg is not None and self.data_cfg is not None)
        self.logger = get_root_logger()
        self.logger.info(self.multi_grid_cfg)

    def before_run(self, runner):
        """Called before running, change the StepLrUpdaterHook to
        RelativeStepLrHook."""
        self._init_schedule(runner, self.multi_grid_cfg, self.data_cfg)
        steps = []
        steps = [s[-1] for s in self.schedule]
        steps.insert(-1, (steps[-2] + steps[-1]) // 2)  # add finetune stage
        for index, hook in enumerate(runner.hooks):
            if isinstance(hook, StepLrUpdaterHook):
                base_lr = hook.base_lr[0]
                gamma = hook.gamma
                lrs = [base_lr * gamma**s[0] * s[1][0] for s in self.schedule]
                lrs = lrs[:-1] + [lrs[-2], lrs[-1] * gamma
                                  ]  # finetune-stage lrs
                new_hook = RelativeStepLrUpdaterHook(runner, steps, lrs)
                runner.hooks[index] = new_hook

    def before_train_epoch(self, runner):
        """Before training epoch, update the runner based on long-cycle
        schedule."""
        self._update_long_cycle(runner)

    def _update_long_cycle(self, runner):
        """Before every epoch, check if long cycle shape should change. If it
        should, change the pipelines accordingly.

        change dataloader and model's subbn3d(split_bn)
        """
        base_b, base_t, base_s = self._get_schedule(runner.epoch)

        # rebuild dataset
        from mmaction.datasets import build_dataset
        resize_list = []
        for trans in self.cfg.data.train.pipeline:
            if trans['type'] == 'SampleFrames':
                curr_t = trans['clip_len']
                trans['clip_len'] = base_t
                trans['frame_interval'] = (curr_t *
                                           trans['frame_interval']) / base_t
            elif trans['type'] == 'Resize':
                resize_list.append(trans)
        resize_list[-1]['scale'] = _ntuple(2)(base_s)

        ds = build_dataset(self.cfg.data.train)

        from mmaction.datasets import build_dataloader

        dataloader = build_dataloader(
            ds,
            self.data_cfg.videos_per_gpu * base_b,
            self.data_cfg.workers_per_gpu,
            dist=True,
            num_gpus=len(self.cfg.gpu_ids),
            drop_last=True,
            seed=self.cfg.get('seed', None),
        )
        runner.data_loader = dataloader
        self.logger.info('Rebuild runner.data_loader')

        # the self._max_epochs is changed, therefore update here
        runner._max_iters = runner._max_epochs * len(runner.data_loader)

        # rebuild all the sub_batch_bn layers
        num_modifies = modify_subbn3d_num_splits(self.logger, runner.model,
                                                 base_b)
        self.logger.info(f'{num_modifies} subbns modified to {base_b}.')

    def _get_long_cycle_schedule(self, runner, cfg):
        # `schedule` is a list of [step_index, base_shape, epochs]
        schedule = []
        avg_bs = []
        all_shapes = []
        self.default_size = self.default_t * self.default_s**2
        for t_factor, s_factor in cfg.long_cycle_factors:
            base_t = int(round(self.default_t * t_factor))
            base_s = int(round(self.default_s * s_factor))
            if cfg.short_cycle:
                shapes = [[
                    base_t,
                    int(round(self.default_s * cfg.short_cycle_factors[0]))
                ],
                          [
                              base_t,
                              int(
                                  round(self.default_s *
                                        cfg.short_cycle_factors[1]))
                          ], [base_t, base_s]]
            else:
                shapes = [[base_t, base_s]]
            # calculate the batchsize, shape = [batchsize, #frames, scale]
            shapes = [[
                int(round(self.default_size / (s[0] * s[1]**2))), s[0], s[1]
            ] for s in shapes]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)

        for hook in runner.hooks:
            if isinstance(hook, LrUpdaterHook):
                if isinstance(hook, StepLrUpdaterHook):
                    steps = hook.step if isinstance(hook.step,
                                                    list) else [hook.step]
                    steps = [0] + steps
                    break
                else:
                    raise NotImplementedError(
                        'Only step scheduler supports multi grid now')
            else:
                pass
        total_iters = 0
        default_iters = steps[-1]
        for step_index in range(len(steps) - 1):
            # except the final step
            step_epochs = steps[step_index + 1] - steps[step_index]
            # number of epochs for this step
            for long_cycle_index, shapes in enumerate(all_shapes):
                cur_epochs = (
                    step_epochs * avg_bs[long_cycle_index] / sum(avg_bs))
                cur_iters = cur_epochs / avg_bs[long_cycle_index]
                total_iters += cur_iters
                schedule.append((step_index, shapes[-1], cur_epochs))
        iter_saving = default_iters / total_iters
        final_step_epochs = runner.max_epochs - steps[-1]
        # the fine-tuning phase to have the same amount of iteration
        # saving as the rest of the training
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]
        # in `schedule` we ignore the shape of ShortCycle
        schedule.append((step_index + 1, all_shapes[-1][-1], ft_epochs))

        x = (
            runner.max_epochs * cfg.epoch_factor / sum(s[-1]
                                                       for s in schedule))
        runner._max_epochs = int(runner._max_epochs * cfg.epoch_factor)
        final_schedule = []
        total_epochs = 0
        for s in schedule:
            # extend the epochs by `factor`
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        self.logger.info(final_schedule)
        return final_schedule

    def _print_schedule(self, schedule):
        """logging the schedule."""
        self.logger.info('\tLongCycleId\tBase shape\tEpochs\t')
        for s in schedule:
            self.logger.info(f'\t{s[0]}\t{s[1]}\t{s[2]}\t')

    def _get_schedule(self, epoch):
        """Returning the corresponding shape."""
        for s in self.schedule:
            if epoch < s[-1]:
                return s[1]
        return self.schedule[-1][1]

    def _init_schedule(self, runner, multi_grid_cfg, data_cfg):
        """Initialize the multigrid shcedule.

        Args:
            runner (:obj: `mmcv.Runner`): The runner within which to train.
            multi_grid_cfg (:obj: `mmcv.ConfigDict`): The multigrid config.
            data_cfg (:obj: `mmcv.ConfigDict`): The data config.
        """
        self.default_bs = data_cfg.videos_per_gpu
        data_cfg = data_cfg.get('train', None)
        final_resize_cfg = [
            aug for aug in data_cfg.pipeline if aug.type == 'Resize'
        ][-1]
        if isinstance(final_resize_cfg.scale, tuple):
            # Assume square image
            if max(final_resize_cfg.scale) == min(final_resize_cfg.scale):
                self.default_s = max(final_resize_cfg.scale)
            else:
                raise NotImplementedError('non-square scale not considered.')
        sample_frame_cfg = [
            aug for aug in data_cfg.pipeline if aug.type == 'SampleFrames'
        ][0]
        self.default_t = sample_frame_cfg.clip_len

        if multi_grid_cfg.long_cycle:
            self.schedule = self._get_long_cycle_schedule(
                runner, multi_grid_cfg)
        else:
            raise ValueError('There should be at least long cycle.')
