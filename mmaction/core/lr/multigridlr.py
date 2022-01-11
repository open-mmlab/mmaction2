from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook


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
