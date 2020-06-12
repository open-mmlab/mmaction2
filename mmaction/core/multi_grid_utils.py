from pdb import set_trace as st

import numpy as np
from mmcv.runner import Hook, LrUpdaterHook
from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook

from mmaction.datasets.pipelines.augmentations import Resize
from mmaction.datasets.pipelines.loading import FrameSelector


class MultiGridHook(Hook):
    """MultiGridHook.
        https://arxiv.org/abs/1912.00998.
    """

    def __init__(self, multi_grid_cfg, data_cfg):
        print(multi_grid_cfg)
        self.multi_grid_cfg = multi_grid_cfg
        self.data_cfg = data_cfg
        self.short_cycle_iter_step = 1
        self.long_cycle_epoch_step = 10

    def before_run(self, runner):
        self._init_schedule(runner, self.multi_grid_cfg, self.data_cfg)

    def before_train_epoch(self, runner):
        print("Before train epoch!")
        if self.every_n_epochs(runner, self.long_cycle_epoch_step):
            print(f'every {self.long_cycle_epoch_step} '
                  f'epochs print it {runner.epoch}')
        if hasattr(runner, 'data_loader'):
            # print("DATALOADER_HERE")
            # for name in vars(runner.data_loader).keys():
            #     print(name)
            # batch_size/dataset

            for name in vars(runner.data_loader.dataset.pipeline).keys():
                print(f'Pipelines {name}')

        else:
            for name in vars(runner).keys():
                print(f"Has attribute {name} in runner")
            for name in vars(self).keys():
                print(f'Has attribute {name} in Hook')

    def before_train_iter(self, runner):
        print(f'now iter: {runner.inner_iter} / {runner.iter}')
        if self.every_n_iters(runner, self.short_cycle_iter_step):
            print(f'every {self.short_cycle_iter_step} '
                  f'iters print it {runner.iter}')

    def _update_long_cycle(self):
        pass

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
                    self.default_s * cfg.short_cycle_factors[0],
                ], [base_t, self.default_s * cfg.short_cycle_factors[1]],
                          [base_t, base_s]]
            else:
                shapes = [[base_t, base_s]]
            # calculate the batchsize, shape = [batchsize, #frames, scale]
            shapes = [[
                int(round(self.default_size / s[0] * s[1]**2)), s[0], s[1]
            ] for s in shapes]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)
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
        for step_index in range(len(steps) - 1):
            # except the final step
            step_epochs = steps[step_index + 1] - steps[step_index]
            # number of epochs for this step
            for long_cycle_index, shapes in enumerate(all_shapes):
                # for long-cycle only, it is [[[8bs, ]],
                # [[4bs,]], [[2bs,]], [[1bs,]]]
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
        '''logging the schedule.
        '''
        print("Long cycle index\tBase shape\tEpochs")
        for s in schedule:
            print("{}\t{}\t{}".format(s[0], s[1], s[2]))

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
