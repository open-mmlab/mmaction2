from abc import ABCMeta, abstractmethod

from mmcv.utils import print_log


class BaseMetrics(metaclass=ABCMeta):

    def __init__(self, logger, **metric_kwargs):
        self.logger = logger
        self.metric_kwargs = metric_kwargs

    @abstractmethod
    def __call__(self, results, gt_labels, kwargs=None):
        """perform evaluation."""

    @abstractmethod
    def wrap_up(self, results):
        """wrap up the results to a dict."""

    def print_log_msg(self, log_msg):
        if self.logger is None:
            return
        msg = ''
        if isinstance(log_msg, str):
            msg = log_msg
        elif isinstance(log_msg, (tuple, list)):
            msg = ''.join(log_msg)
        elif isinstance(log_msg, dict):
            for k, v in log_msg.items():
                msg += f'\n{k}\t{v:.4f}'
        print_log(msg, logger=self.logger)
