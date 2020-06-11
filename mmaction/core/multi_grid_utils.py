from mmcv.runner import Hook


class MultiGridHook(Hook):
    """MultiGridHook.
    """

    def __init__(self, multi_grid_cfg, data_cfg):
        print(multi_grid_cfg)
        self.short_cycle_iter_step = 1
        self.long_cycle_epoch_step = 10
        self._init_schedule()

    def before_run(self, runner):
        print("Before run hook!")
        if hasattr(runner, 'data_loader'):
            print(type(runner.data_loader))

    def before_train_epoch(self, runner):
        print("Before train epoch!")

    def before_train_iter(self, runner):
        print(f'now iter: {runner.inner_iter} / {runner.iter}')
        if self.every_n_iters(runner, self.short_cycle_iter_step):
            print(f'every {self.short_cycle_iter_step} '
                  f'iters print it {runner.iter}')

    def _update_long_cycle(self):
        pass

    def _get_schedule(self):
        pass

    def _init_schedule(self):
        pass
