# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mmaction.datasets.samplers import (ClassSpecificDistributedSampler,
                                        DistributedInfiniteGroupSampler,
                                        DistributedInfiniteSampler,
                                        DistributedSampler,
                                        InfiniteGroupSampler, InfiniteSampler)


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


class ExampleDataset(Dataset):

    def __init__(self):
        self.flag = np.array([0, 1], dtype=np.uint8)

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([idx]), img_metas=dict(idx=idx))
        return results

    def __len__(self):
        return 2


class ExampleDataset2(Dataset):

    def __init__(self):
        self.flag = np.array([0, 1, 1, 1], dtype=np.uint8)

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([idx]), img_metas=dict(idx=idx))
        return results

    def __len__(self):
        return 4


def test_infinite_sampler():
    dataset = ExampleDataset()
    sampler = InfiniteSampler(dataset=dataset, shuffle=False)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=1)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert 'img' in data
        assert 'img_metas' in data


def test_infinite_group_sampler():
    dataset = ExampleDataset()
    sampler = InfiniteGroupSampler(
        dataset=dataset, shuffle=False, samples_per_gpu=2)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=2)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert torch.allclose(data['img_metas']['idx'][0],
                              data['img_metas']['idx'][1])


def test_dist_infinite_sampler():
    dataset = ExampleDataset()
    sampler = DistributedInfiniteSampler(
        dataset=dataset, shuffle=False, num_replicas=2, rank=0)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=1)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert data['img'].item() == 0


def test_dist_group_infinite_sampler():
    dataset = ExampleDataset2()
    sampler = DistributedInfiniteGroupSampler(
        dataset=dataset,
        shuffle=False,
        num_replicas=2,
        rank=0,
        samples_per_gpu=2)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=2)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        if i % 2 == 0:
            assert torch.allclose(data['img_metas']['idx'],
                                  torch.tensor([0, 0]))
        else:
            assert torch.allclose(data['img_metas']['idx'],
                                  torch.tensor([2, 2]))
    sampler = DistributedInfiniteGroupSampler(
        dataset=dataset,
        shuffle=False,
        num_replicas=2,
        rank=1,
        samples_per_gpu=2)
    dataloader = DataLoader(
        dataset=dataset, num_workers=0, sampler=sampler, batch_size=2)
    dataloader_iter = iter(dataloader)
    for i in range(5):
        data = next(dataloader_iter)
        assert torch.allclose(data['img_metas']['idx'], torch.tensor([1, 3]))
