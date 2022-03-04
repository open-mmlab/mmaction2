# Copyright (c) OpenMMLab. All rights reserved.
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

    def __init__(self,
                 runner,
                 steps,
                 lrs,
                 warmup_epochs=34,
                 warmuplr_start=0.01,
                 **kwargs):
        super().__init__(**kwargs)
        assert len(steps) == (len(lrs))
        self.steps = steps
        self.lrs = lrs
        self.warmup_epochs = warmup_epochs
        self.warmuplr_start = warmuplr_start
        self.warmuplr_end = self.lrs[0]
        super().before_run(runner)

    def get_lr(self, runner, base_lr):
        """Similar to that of mmcv."""
        progress = runner.epoch if self.by_epoch else runner.iter
        if progress <= self.warmup_epochs:
            alpha = (self.warmuplr_end -
                     self.warmuplr_start) / self.warmup_epochs
            return progress * alpha + self.warmuplr_start
        for i in range(len(self.steps)):
            if progress < self.steps[i]:
                return self.lrs[i]
