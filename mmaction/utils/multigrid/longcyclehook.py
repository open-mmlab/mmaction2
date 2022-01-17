import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook, StepLrUpdaterHook

from mmaction.core.lr import RelativeStepLrUpdaterHook
from mmaction.utils import get_root_logger


@HOOKS.register_module()
class LongCycleHook(Hook):
    """A multigrid hook without subbn, preicsebn.

    multi_grid = dict(
        long_cycle=True,
        short_cycle=True,
        long_cycle_factors=((0.25,0.5**0.5),(0.5,0.5**0.5),(0.5,1),(1,1)),
        short_cycle_factors=(0.5,0.5**0.5),
        epoch_factor=1.5,
        default_s=(224, 224))
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
                self.logger.info(f'lrs: {lrs}, steps: {steps}')
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
        from mmaction.datasets import build_dataset
        for trans in self.cfg.data.train.pipeline:
            if trans['type'] == 'SampleFrames':
                curr_t = trans['clip_len']
                trans['clip_len'] = base_t
                trans['frame_interval'] = (curr_t *
                                           trans['frame_interval']) / base_t
                self.logger.info(trans)
        for trans in self.cfg.data.train.pipeline[::-1]:
            if trans['type'] == 'Resize':
                trans['scale'] = (base_s, base_s)
                self.logger.info(trans)
                break
        ds = build_dataset(self.cfg.data.train)
        self.logger.info('rebuild dataset(train)--before train epoch')

        from mmaction.datasets import build_dataloader
        dataloader = build_dataloader(
            ds,
            self.data_cfg.videos_per_gpu * base_b,
            self.data_cfg.workers_per_gpu,
            dist=self.cfg.get('dist', True),
            num_gpus=len(self.cfg.gpu_ids),
            drop_last=True,
            seed=self.cfg.get('seed', None),
        )
        runner.data_loader = dataloader

        # the self._max_epochs is changed, therefore update here
        runner._max_iters = runner._max_epochs * len(runner.data_loader)

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
                # shapes = [[base_t, base_s]]

            else:
                shapes = [[base_t, base_s]]
            # calculate the batchsize, shape = [batchsize, #frames, scale]
            shapes = [[
                int(round(self.default_size / (s[0] * s[1]**2))), s[0], s[1]
            ] for s in shapes]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)
        # print('all-shapes---', all_shapes)
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
        '''
        [(0, [8, 8, 158], 73), (0, [4, 16, 158], 110), (0, [2, 16, 224], 142),
        (0, [1, 32, 224], 158),(1, [8, 8, 158], 205), (1, [4, 16, 158], 228),
        (1, [2, 16, 224], 248), (1, [1, 32, 224], 259),(2, [8, 8, 158], 291),
        (2, [4, 16, 158], 308), (2, [2, 16, 224], 322), (2, [1, 32, 224], 329),
        (3, [1, 32, 224], 358)]
        '''

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

            self.change_epochs = [0]
            for sche in self.schedule:
                self.change_epochs.append(sche[-1])
            print('self.change_epochs---', self.change_epochs)

        else:
            raise ValueError('There should be at least long cycle.')
