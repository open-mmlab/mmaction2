import torch
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
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


class DistributedPowerSampler(_DistributedSampler):
    """DistributedPowerSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    Samples are sampled with the probability that is proportional to the power
    of label frequency (freq ^ power). The sampler only applies to single class
    recognition dataset.

    The default value of power is 1, which is equivalent to bootstrap sampling
    from the entire dataset.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, power=1):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.power = power

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        video_infos_by_class = self.dataset.video_infos_by_class
        num_classes = self.dataset.num_classes
        # For simplicity, discontinuous labels are not permitted
        assert set(video_infos_by_class) == set(range(num_classes))
        counts = [len(video_infos_by_class[i]) for i in range(num_classes)]
        counts = [cnt**self.power for cnt in counts]

        indices = torch.multinomial(
            torch.Tensor(counts),
            self.total_size,
            replacement=True,
            generator=g)
        indices = indices.data.numpy().tolist()
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
