from mmcv.utils import build_from_cfg

from .registry import METRICS


def build_metrics(cfg):

    return build_from_cfg(cfg, METRICS)
