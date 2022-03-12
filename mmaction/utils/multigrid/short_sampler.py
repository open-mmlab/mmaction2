# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from torch.utils.data.sampler import Sampler


class ShortCycleSampler(Sampler):
    """Extend Sampler to support "short cycle" sampling.

    See paper "A Multigrid Method for Efficiently Training Video Models", Wu et
    al., 2019 (https://arxiv.org/abs/1912.00998) for details.

    Args:
        sampler (:obj: `torch.Sampler`): The default sampler to be warpped.
        batch_size (int): The batchsize before short-cycle modification.
        multi_grid_cfg (dict): The config dict for multigrid training.
        crop_size (int): The actual spatial scale.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: True.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 multigrid_cfg,
                 crop_size,
                 drop_last=True):

        self.sampler = sampler
        self.drop_last = drop_last

        bs_factor = [
            int(
                round(
                    (float(crop_size) / (s * multigrid_cfg.default_s[0]))**2))
            for s in multigrid_cfg.short_cycle_factors
        ]

        self.batch_sizes = [
            batch_size * bs_factor[0], batch_size * bs_factor[1], batch_size
        ]

    def __iter__(self):
        counter = 0
        batch_size = self.batch_sizes[0]
        batch = []
        for idx in self.sampler:
            batch.append((idx, counter % 3))
            if len(batch) == batch_size:
                yield batch
                counter += 1
                batch_size = self.batch_sizes[counter % 3]
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        avg_batch_size = sum(self.batch_sizes) / 3.0
        if self.drop_last:
            return int(np.floor(len(self.sampler) / avg_batch_size))
        else:
            return int(np.ceil(len(self.sampler) / avg_batch_size))
