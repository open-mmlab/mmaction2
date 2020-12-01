import numpy as np
from torch.utils.data.sampler import Sampler


class ShortCycleBatchSampler(Sampler):
    """Extend Sampler to support "short cycle" sampling. The Sampler input can
    be both distributed or non-distributed. See paper `A Multigrid Method for
    Efficiently Training Video Models.

        <https://arxiv.org/abs/1912.00998>`_ for details.
    Args:
        sampler (:obj: `torch.Sampler`): The default sampler to be warpped.
        batch_size (int): The batchsize before short-cycle modification.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
        multi_grid_cfg (dict): The config dict for multi-grid training.
        crop_size (int): The actual spatial scale.
    """

    def __init__(self, sampler, batch_size, drop_last, multi_grid_cfg,
                 crop_size):
        self.sampler = sampler
        self.drop_last = drop_last

        bs_factor = [
            int(round(1 / s**2)) for s in multi_grid_cfg.short_cycle_factors
        ]

        bs_factor = [
            int(
                round(
                    (float(crop_size) / (s * multi_grid_cfg.default_s[0]))**2))
            for s in multi_grid_cfg.short_cycle_factors
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
