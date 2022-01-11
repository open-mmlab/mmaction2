import math

import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler


class DistributedShortCycleSampler(BatchSampler):

    def __init__(self,
                 dataset,
                 batch_sizes,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 drop_last=True):
        '''
        batch_sizes = [
            batch_size * bs_factor[0],
            batch_size * bs_factor[1],
            batch_size,
        ]
        '''
        assert any(
            isinstance(batch_size, int) and batch_size > 0 for batch_size in
            batch_sizes), 'batch_size should be a positive integer'

        self.dataset = dataset
        self.batch_sizes = batch_sizes
        print('self.batch_sizes in sampler---', self.batch_sizes)
        self.len_batch_sizes = len(self.batch_sizes)

        assert isinstance(shuffle, bool), \
            'shuffle should be a boolean value'
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
            'drop_last should be a boolean number'

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError('Invalid rank {}, rank should be in the interval'
                             ' [0, {}]'.format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            total_batch_size = sum(self.batch_sizes)
            subsampled_indices = []
            # number samples of last batch
            last_batch_size = self.total_size % (
                total_batch_size * self.num_replicas)
            assert last_batch_size % self.num_replicas == 0
            last_local_batch_size = last_batch_size // self.num_replicas

            for i in range(self.rank * total_batch_size,
                           len(indices) - last_batch_size,
                           total_batch_size * self.num_replicas):
                subsampled_indices.extend(indices[i:i + total_batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(
                indices[self.rank * last_local_batch_size:(self.rank + 1) *
                        last_local_batch_size])
            return subsampled_indices

        if self.num_replicas > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        counter = 0
        batch_size = self.batch_sizes[0]
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == batch_size:
                # print(self.rank, 'batch-indices---', batch_indices)
                yield batch_indices
                counter += 1
                batch_size = self.batch_sizes[counter % self.len_batch_sizes]
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        avg_batch_size = sum(self.batch_sizes) / float(self.len_batch_sizes)
        if self.drop_last:
            return int(np.floor(self.num_samples / avg_batch_size))
        else:
            return int(np.ceil(self.num_samples / avg_batch_size))

    def set_epoch(self, epoch):
        self.epoch = epoch
