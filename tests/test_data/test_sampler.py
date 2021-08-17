# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import DataLoader, Dataset

from mmaction.datasets.samplers import (ClassSpecificDistributedSampler,
                                        DistributedSampler)


class MyDataset(Dataset):

    def __init__(self, class_prob={i: 1 for i in range(10)}):
        super().__init__()
        self.class_prob = class_prob
        self.video_infos = [
            dict(data=idx, label=idx % 10) for idx in range(100)
        ]

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        return self.video_infos[idx]


def test_distributed_sampler():
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for _, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 25
    assert sum([len(x['data']) for x in batches]) == 100

    sampler = DistributedSampler(dataset, num_replicas=4, rank=2)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for i, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 7
    assert sum([len(x['data']) for x in batches]) == 25

    sampler = DistributedSampler(dataset, num_replicas=6, rank=3)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for i, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 5
    assert sum([len(x['data']) for x in batches]) == 17


def test_class_specific_distributed_sampler():
    class_prob = dict(zip(list(range(10)), [1] * 5 + [3] * 5))
    dataset = MyDataset(class_prob=class_prob)

    sampler = ClassSpecificDistributedSampler(
        dataset, num_replicas=1, rank=0, dynamic_length=True)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for _, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 50
    assert sum([len(x['data']) for x in batches]) == 200

    sampler = ClassSpecificDistributedSampler(
        dataset, num_replicas=1, rank=0, dynamic_length=False)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for i, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 25
    assert sum([len(x['data']) for x in batches]) == 100

    sampler = ClassSpecificDistributedSampler(
        dataset, num_replicas=6, rank=2, dynamic_length=True)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for i, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 9
    assert sum([len(x['data']) for x in batches]) == 34

    sampler = ClassSpecificDistributedSampler(
        dataset, num_replicas=6, rank=2, dynamic_length=False)
    data_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    batches = []
    for i, data in enumerate(data_loader):
        batches.append(data)

    assert len(batches) == 5
    assert sum([len(x['data']) for x in batches]) == 17
