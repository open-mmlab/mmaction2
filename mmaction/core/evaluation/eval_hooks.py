import os.path as osp
import warnings
from math import inf

import mmcv
from mmcv.runner import Hook
from torch.utils.data import DataLoader

from mmaction.utils import get_root_logger


class EpochEvalHook(Hook):
    """Non-Distributed evaluation hook based on epochs.

    Notes:
        If new arguments are added for EpochEvalHook, tools/test.py,
        tools/eval_metric.py may be effected.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        save_best (bool): Whether to save best checkpoint during evaluation.
            Default: True.
        key_indicator (str | None): Key indicator to measure the best
            checkpoint during evaluation when ``save_best`` is set to True.
            Options are the evaluation metrics to the test dataset. e.g.,
             ``top1_acc``, ``top5_acc``, ``mean_class_accuracy``,
            ``mean_average_precision``, ``mmit_mean_average_precision``
            for action recognition dataset (RawframeDataset and VideoDataset).
            ``AR@AN``, ``auc`` for action localization dataset
            (ActivityNetDataset). Default: `top1_acc`.
        rule (str | None): Comparison rule for best score. Options are None,
            'greater' and 'less'. If set to None, it will infer a reasonable
            rule. Default: 'None'.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision']
    less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 save_best=True,
                 key_indicator='top1_acc',
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')
        if not isinstance(save_best, bool):
            raise TypeError("'save_best' should be a boolean")

        if save_best and not key_indicator:
            raise ValueError('key_indicator should not be None, when '
                             'save_best is set to True.')
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None and save_best:
            if any(key in key_indicator for key in self.greater_keys):
                rule = 'greater'
            elif any(key in key_indicator for key in self.less_keys):
                rule = 'less'
            else:
                raise ValueError(
                    f'key_indicator must be in {self.greater_keys} '
                    f'or in {self.less_keys} when rule is None, '
                    f'but got {key_indicator}')

        if interval <= 0:
            raise ValueError(f'interval must be positive, but got {interval}')
        if start is not None and start < 0:
            warnings.warn(
                f'The evaluation start epoch {start} is smaller than 0, '
                f'use 0 instead', UserWarning)
            start = 0

        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.eval_kwargs = eval_kwargs
        self.save_best = save_best
        self.key_indicator = key_indicator
        self.rule = rule

        self.logger = get_root_logger()

        if self.save_best:
            self.compare_func = self.rule_map[self.rule]
            self.best_score = self.init_value_map[self.rule]

        self.best_json = dict()
        self.initial_epoch_flag = True

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training."""
        if not self.initial_epoch_flag:
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_epoch_flag = False

    def evaluation_flag(self, runner):
        """Judge whether to perform_evaluation after this epoch.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                # No evaluation during the interval epochs.
                return False
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False
        return True

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if not self.evaluation_flag(runner):
            return

        current_ckpt_path = osp.join(runner.work_dir,
                                     f'epoch_{runner.epoch + 1}.pth')
        json_path = osp.join(runner.work_dir, 'best.json')

        if osp.exists(json_path) and len(self.best_json) == 0:
            self.best_json = mmcv.load(json_path)
            self.best_score = self.best_json['best_score']
            self.best_ckpt = self.best_json['best_ckpt']
            self.key_indicator = self.best_json['key_indicator']

        from mmaction.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader)
        key_score = self.evaluate(runner, results)
        if (self.save_best and self.compare_func(key_score, self.best_score)):
            self.best_score = key_score
            self.logger.info(
                f'Now best checkpoint is epoch_{runner.epoch + 1}.pth')
            self.best_json['best_score'] = self.best_score
            self.best_json['best_ckpt'] = current_ckpt_path
            self.best_json['key_indicator'] = self.key_indicator
            mmcv.dump(self.best_json, json_path)

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.key_indicator is not None:
            if self.key_indicator not in eval_res:
                warnings.warn('The key indicator for evaluation is not '
                              'included in evaluation result, please specify '
                              'it in config file')
                return None
            return eval_res[self.key_indicator]

        return None


class DistEpochEvalHook(EpochEvalHook):
    """Distributed evaluation hook based on epochs.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        save_best (bool): Whether to save best checkpoint during evaluation.
            Default: True.
        key_indicator (str | None): Key indicator to measure the best
            checkpoint during evaluation when ``save_best`` is set to True.
            Options are the evaluation metrics to the test dataset. e.g.,
             ``top1_acc``, ``top5_acc``, ``mean_class_accuracy``,
            ``mean_average_precision``, ``mmit_mean_average_precision``
            for action recognition dataset (RawframeDataset and VideoDataset).
            ``AR@AN``, ``auc`` for action localization dataset
            (ActivityNetDataset). Default: `top1_acc`.
        rule (str | None): Comparison rule for best score. Options are None,
            'greater' and 'less'. If set to None, it will infer a reasonable
            rule. Default: 'None'.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 save_best=True,
                 key_indicator='top1_acc',
                 rule=None,
                 tmpdir=None,
                 gpu_collect=False,
                 **eval_kwargs):
        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            save_best=save_best,
            key_indicator=key_indicator,
            rule=rule,
            **eval_kwargs)
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def after_train_epoch(self, runner):
        """Called after each training epoch to evaluate the model."""
        if not self.evaluation_flag(runner):
            return

        current_ckpt_path = osp.join(runner.work_dir,
                                     f'epoch_{runner.epoch + 1}.pth')
        json_path = osp.join(runner.work_dir, 'best.json')

        if osp.exists(json_path) and len(self.best_json) == 0:
            self.best_json = mmcv.load(json_path)
            self.best_score = self.best_json['best_score']
            self.best_ckpt = self.best_json['best_ckpt']
            self.key_indicator = self.best_json['key_indicator']

        from mmaction.apis import multi_gpu_test

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if (self.save_best and key_score is not None
                    and self.compare_func(key_score, self.best_score)):
                self.best_score = key_score
                self.logger.info(
                    f'Now best checkpoint is epoch_{runner.epoch + 1}.pth')
                self.best_json['best_score'] = self.best_score
                self.best_json['best_ckpt'] = current_ckpt_path
                self.best_json['key_indicator'] = self.key_indicator
                mmcv.dump(self.best_json, json_path)
