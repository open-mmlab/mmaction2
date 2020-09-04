import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import HOOKS, Hook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook, StepLrUpdaterHook
from torch.nn.modules.utils import _ntuple

from mmaction.datasets.pipelines import Resize, SampleFrames
from ..utils import get_root_logger


def modify_subbn3d_num_splits(logger, module, num_splits):
    """Recursively modify the number of splits of subbn3ds in module.

    Inheritates the running_mean and running_var from last split_bn, while
        spares the num_batch_tracked. This operation may result in incorrect
        calculation of mean and var by EMA.

    Args:
        logger (:obj:`logging.Logger`): The logger to log infomation.
        module (nn.Module): The module to be modified.
        num_splits (int): The targeted number of splits.

    Returns:
        int: The number of subbn3d modules modified.
    """
    count = 0
    for child in module.children():
        from mmaction.models import SubBatchBN3d
        if isinstance(child, SubBatchBN3d):
            new_split_bn = nn.BatchNorm3d(
                child.num_features * num_splits, affine=False).cuda()
            new_state_dict = new_split_bn.state_dict()
            for param_name, param in child.split_bn.state_dict().items():
                origin_param_shape = param.size()
                new_param_shape = new_state_dict[param_name].size()
                if (len(origin_param_shape) == 1 and len(new_param_shape) == 1
                        and new_param_shape[0] > origin_param_shape[0]
                        and new_param_shape[0] % origin_param_shape[0] == 0):
                    new_state_dict[param_name] = torch.cat(
                        [param] *
                        (new_param_shape[0] // origin_param_shape[0]))
                    logger.info(f'{param_name} modified to {new_param_shape}')
                else:
                    logger.info(f'skip {param_name}')
            child.num_splits = num_splits
            new_split_bn.load_state_dict(new_state_dict)
            child.split_bn = new_split_bn
            count += 1
        else:
            count += modify_subbn3d_num_splits(logger, child, num_splits)
    return count


@HOOKS.register_module()
class RelativeStepLrUpdaterHook(LrUpdaterHook):
    """RelativeStepLrUpdaterHook.

    Args:
        runner (:obj:`mmcv.Runner`): The runner instance used.
        steps (list[int]): The list of epochs at which decrease
            the learning rate.
        **kwargs (dict): Same as that of mmcv.
    """

    def __init__(self, runner, steps, lrs, **kwargs):
        super().__init__(**kwargs)
        assert len(steps) == (len(lrs))
        self.steps = steps
        self.lrs = lrs
        super().before_run(runner)

    def get_lr(self, runner, base_lr):
        """Similar to that of mmcv."""
        progress = runner.epoch if self.by_epoch else runner.iter
        for i in range(len(self.steps)):
            if progress < self.steps[i]:
                return self.lrs[i]


class MultiGridHook(Hook):
    """A multigrid method for efficiently training video models.

        This hook defines multigrid training schedule and update cfg
        accordingly, which is proposed in `A Multigrid Method for Efficiently
        Training Video Models <https://arxiv.org/abs/1912.00998>`_.

    Args:
        cfg (:obj:`mmcv.ConfigDictg`): The whole config for the experiment.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.multi_grid_cfg = cfg.get('multi_grid', None)
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
                self.logger.info(f'lrs: {lrs}, steps: {steps}')
                new_hook = RelativeStepLrUpdaterHook(runner, steps, lrs)
                runner.hooks[index] = new_hook

    def before_train_epoch(self, runner):
        """Before training epoch, update the runner based on long-cycle
        schedule."""
        self._update_long_cycle(runner)

    def after_train_epoch(self, runner):
        """After training epoch, aggregate the stats from split_bns to bns."""
        from mmaction.models.common import aggregate_sub_bn_stats
        num_sub_bn3d_aggregated = aggregate_sub_bn_stats(runner.model)
        self.logger.info(f'{num_sub_bn3d_aggregated} aggregated.')

    def _update_long_cycle(self, runner):
        """Before every epoch, check if long cycle shape should change.

        If it should, change the pipelines accordingly.
        """
        modified = False
        base_b, base_t, base_s = self._get_schedule(runner.epoch)
        resize_list = []  # use a list to find the final `Resize`
        for trans in runner.data_loader.dataset.pipeline.transforms:
            if isinstance(trans, Resize):
                resize_list.append(trans)
            elif isinstance(trans, SampleFrames):
                curr_t = trans.clip_len
                if base_t != curr_t:
                    # Change the T-dimension
                    modified = True
                    trans.clip_len = base_t
                    trans.frame_interval = (curr_t *
                                            trans.frame_interval) / base_t
        curr_s = min(resize_list[-1].scale)  # Assume it's square
        if curr_s != base_s:
            modified = True
            # Change the S-dimension
            resize_list[-1].scale = _ntuple(2)(base_s)

        # swap the dataloader with a new one
        ds = getattr(runner.data_loader, 'dataset')
        from mmaction.datasets import build_dataloader
        dataloader = build_dataloader(
            ds,
            self.data_cfg.videos_per_gpu * base_b,
            self.data_cfg.workers_per_gpu,
            dist=self.cfg.get('dist', True),
            drop_last=self.data_cfg.get('train_drop_last', True),
            seed=self.cfg.get('seed', None),
            short_cycle=self.multi_grid_cfg.short_cycle,
            multi_grid_cfg=self.multi_grid_cfg,
            crop_size=base_s)
        runner.data_loader = dataloader

        # rebuild all the sub_batch_bn layers
        if modified:
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
                # shape = [#frames, scale]
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
        final_schedule = []
        total_epochs = 0
        for s in schedule:
            # extend the epochs by `factor`
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        self._print_schedule(final_schedule)
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
        """Initialize the multi-grid shcedule.

        Args:
            runner (:obj: `mmcv.Runner`): The runner within which to train.
            multi_grid_cfg (:obj: `mmcv.ConfigDict`): The multi-grid config.
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
