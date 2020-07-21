import numpy as np
from mmcv.cnn import build_norm_layer
from mmcv.runner import HOOKS, Hook, LrUpdaterHook
from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook
from torch.nn.modules.utils import _ntuple

from mmaction.datasets.builder import build_dataloader
from mmaction.datasets.pipelines.augmentations import Resize
from mmaction.datasets.pipelines.loading import SampleFrames
from mmaction.models.common import SubBatchBN3d
from ..utils import get_root_logger


@HOOKS.register_module()
class FixedStepwiseLrUpdaterHook(StepLrUpdaterHook):
    """StepLrUpdater that change lr with an extra factor ``step_lr_ratio``.

    Args:
        ori_step (list[int]): Same as that of mmcv.
        new_step (list[int]): The real steps caused by multi-grid
            training.
        step_lr_ratio (list[int]): The extra factor to be multiplied
            at each new step.
        gamma (float): Same as that of mmcv.
        **kwargs (dict): Same as that of mmcv.
    """

    def __init__(self, ori_step, new_step, step_lr_ratio, gamma=0.1, **kwargs):
        super().__init__(ori_step, gamma=gamma, **kwargs)
        self.step_lr_ratio = step_lr_ratio
        self.new_step = new_step

    def get_lr(self, runner, base_lr):
        progress = runner.epoch
        exp = len(self.step)
        lr_ratio = 1.0
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        for new_i, new_s in enumerate(self.new_step):
            if progress < new_s:
                lr_ratio = self.step_lr_ratio[new_i]
                break
        return base_lr * self.gamma**exp * lr_ratio


class MultiGridHook(Hook):
    """MultiGridHook.

    https://arxiv.org/abs/1912.00998.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.multi_grid_cfg = cfg.get('multi_grid', None)
        self.data_cfg = cfg.get('data', None)
        assert (self.multi_grid_cfg is not None and self.data_cfg is not None)
        self.logger = get_root_logger()
        self.logger.info(self.multi_grid_cfg)

    def before_run(self, runner):
        self._init_schedule(runner, self.multi_grid_cfg, self.data_cfg)
        step = []
        step = [s[-1] for s in self.schedule]
        step[-1] = (step[-2] + step[-1]) // 2
        for hook in runner.hooks:
            if isinstance(hook, StepLrUpdaterHook):
                ori_step = hook.step
                step_lr_ratio = [s[1][0] for s in self.schedule]
                print(f'step_lr_ratio  {step_lr_ratio} '
                      f'ori_step {ori_step} new_step {step}')
                new_hook = FixedStepwiseLrUpdaterHook(ori_step, step,
                                                      step_lr_ratio)
                runner.register_hook(new_hook)
                runner.hooks.remove(hook)

    def before_train_epoch(self, runner):
        self._update_long_cycle(runner)

    def _update_long_cycle(self, runner):
        """Before every epoch, check if long cycle shape should change.

        If it should, change the runner's optimizer and pipelines accordingly.
        """
        base_b, base_t, base_s = self._get_schedule(runner.epoch)
        resize_list = []  # use a list to find the final `Resize`
        for trans in runner.data_loader.dataset.pipeline.transforms:
            if isinstance(trans, Resize):
                resize_list.append(trans)
            elif isinstance(trans, SampleFrames):
                curr_t = trans.clip_len
                if base_t != curr_t:
                    # Change the T-dimension
                    trans.clip_len = base_t
                    trans.frame_interval = (curr_t *
                                            trans.frame_interval) / base_t
        curr_s = min(resize_list[-1].scale)  # Assume it's square
        if curr_s != base_s:
            # Change the S-dimension
            resize_list[-1].scale = _ntuple(2)(base_s)

        # swap the dataloader with a new one
        ds = getattr(runner.data_loader, 'dataset')
        dataloader = build_dataloader(
            ds,
            self.data_cfg.videos_per_gpu * base_b,  # change here
            self.data_cfg.workers_per_gpu,
            dist=True,
            drop_last=self.data_cfg.get('train_drop_last', False),
            seed=self.cfg.seed)
        runner.data_loader = dataloader

        # rebuild all the sub_batch_bn layers
        if curr_t != base_t or curr_s != base_s:
            num_modifies = self.modify_num_splits(runner.model, base_b)
            self.logger.info(f'{num_modifies} subbns modified.')

    def modify_num_splits(self, module, base_b):
        count = 0
        for child in module.children():
            if isinstance(child, SubBatchBN3d):
                child = build_norm_layer(
                    dict(type='SubBatchBN3d', num_splits=base_b),
                    child.num_features)
                count += 1
            else:
                count += self.modify_num_splits(child, base_b)
        return count

    def _get_long_cycle_schedule(self, runner, cfg):
        # schedule is a list of [step_index, base_shape, epochs]
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
                    int(round(self.default_s * cfg.short_cycle_factors[0])),
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
        self.logger.info(f'all shapes are {all_shapes}')
        for hook in runner.hooks:
            # search for `steps`, maybe a better way is to read
            # the cfg of optimizer
            if isinstance(hook, LrUpdaterHook):
                if isinstance(hook, StepLrUpdaterHook):
                    steps = hook.step if isinstance(hook.step,
                                                    list) else [hook.step]
                    break
                else:
                    raise NotImplementedError(
                        'Only step scheduler supports multi grid now')
            else:
                pass
        total_iters = 0
        default_iters = steps[-1]
        steps = [0] + steps
        for step_index in range((len(steps) - 1)):
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
        # saving as the rest of the training.
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]
        schedule.append((step_index + 1, all_shapes[-1][2], ft_epochs))
        # Obtrain final schedule given desired cfg.MULTIGRID.EPOCH_FACTOR.
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
            self.logger.info('\t{}\t{}\t{}\t'.format(s[0], s[1], s[2]))

    def _get_schedule(self, epoch):
        """Returning the corresponding shape."""
        for s in self.schedule:
            if epoch < s[-1]:
                return s[1]
        return self.schedule[-1][1]

    def _init_schedule(self, runner, multi_grid_cfg, data_cfg):
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
        self.default_span = (
            sample_frame_cfg.clip_len * sample_frame_cfg.frame_interval)
        if multi_grid_cfg.long_cycle:
            self.schedule = self._get_long_cycle_schedule(
                runner, multi_grid_cfg)
        else:
            raise ValueError('There should be at lease long cycle.')
