from mmcv.utils import build_from_cfg

from .dataset_wrappers import RepeatDataset
from .registry import DATASETS


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset
