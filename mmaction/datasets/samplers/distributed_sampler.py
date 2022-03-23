# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import defaultdict

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmaction.core import sync_random_seed


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        # In distributed sampling, different ranks should sample non-overlapped
        # data in the dataset. Therefore, this function is used to make sure
        # that each rank shuffles the data indices in the same order based
        # on the same seed. Then different ranks could use different indices
        # to select non-overlapped data from the same data list.
        self.seed = sync_random_seed(seed) if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    Samples are sampled with a class specific probability, which should be an
    attribute of the dataset (dataset.class_prob, which is a dictionary that
    map label index to the prob). This sampler is only applicable to single
    class recognition dataset. This sampler is also compatible with
    RepeatDataset.

    The default value of dynamic_length is True, which means we use
    oversampling / subsampling, and the dataset length may changed. If
    dynamic_length is set as False, the dataset length is fixed.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 dynamic_length=True,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

        if type(dataset).__name__ == 'RepeatDataset':
            dataset = dataset.dataset

        assert hasattr(dataset, 'class_prob')

        self.class_prob = dataset.class_prob
        self.dynamic_length = dynamic_length
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        class_indices = defaultdict(list)

        # To be compatible with RepeatDataset
        times = 1
        dataset = self.dataset
        if type(dataset).__name__ == 'RepeatDataset':
            times = dataset.times
            dataset = dataset.dataset
        for i, item in enumerate(dataset.video_infos):
            class_indices[item['label']].append(i)

        if self.dynamic_length:
            indices = []
            for k, prob in self.class_prob.items():
                prob = prob * times
                for i in range(int(prob // 1)):
                    indices.extend(class_indices[k])
                rem = int((prob % 1) * len(class_indices[k]))
                rem_indices = torch.randperm(
                    len(class_indices[k]), generator=g).tolist()[:rem]
                indices.extend(rem_indices)
            if self.shuffle:
                shuffle = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in shuffle]

            # re-calc num_samples & total_size
            self.num_samples = math.ceil(len(indices) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # We want to keep the dataloader length same as original
            video_labels = [x['label'] for x in dataset.video_infos]
            probs = [
                self.class_prob[lb] / len(class_indices[lb])
                for lb in video_labels
            ]

            indices = torch.multinomial(
                torch.Tensor(probs),
                self.total_size,
                replacement=True,
                generator=g)
            indices = indices.data.numpy().tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # retrieve indices for current process
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
