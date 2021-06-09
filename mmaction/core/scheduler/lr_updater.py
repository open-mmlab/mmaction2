from mmcv.runner import HOOKS, LrUpdaterHook
from mmcv.runner.hooks.lr_updater import annealing_cos


@HOOKS.register_module()
class TINLrUpdaterHook(LrUpdaterHook):

    def __init__(self, min_lr, **kwargs):
        self.min_lr = min_lr
        super().__init__(**kwargs)

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'linear':
            # 'linear' warmup is rewritten according to TIN repo:
            # https://github.com/deepcs233/TIN/blob/master/main.py#L409-L412
            k = (cur_iters / self.warmup_iters) * (
                1 - self.warmup_ratio) + self.warmup_ratio
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        elif self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        target_lr = self.min_lr
        if self.warmup is not None:
            progress = progress - self.warmup_iters
            max_progress = max_progress - self.warmup_iters
        factor = progress / max_progress
        return annealing_cos(base_lr, target_lr, factor)
