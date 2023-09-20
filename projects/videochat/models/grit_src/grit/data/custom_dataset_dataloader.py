# Copyright (c) Facebook, Inc. and its affiliates. Modified by Jialian Wu
# from https://github.com/facebookresearch/Detic/blob/main/detic/data
# /custom_dataset_dataloader.py
import itertools
import operator
from typing import Optional

import torch
import torch.utils.data
from detectron2.config import configurable
from detectron2.data.build import (check_metadata_consistency,
                                   filter_images_with_few_keypoints,
                                   filter_images_with_only_crowd_annotations,
                                   get_detection_dataset_dicts,
                                   print_instances_class_histogram,
                                   worker_init_reset_seed)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from torch.utils.data.sampler import Sampler


def _custom_train_loader_from_config(cfg,
                                     mapper=None,
                                     *,
                                     dataset=None,
                                     sampler=None):
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if 'MultiDataset' in sampler_name:
        dataset_dicts = get_detection_dataset_dicts_with_source(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS else None,
        )
    else:
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is not None:
        pass
    elif sampler_name == 'TrainingSampler':
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == 'MultiDatasetSampler':
        sampler = MultiDatasetSampler(
            dataset_dicts,
            dataset_ratio=cfg.DATALOADER.DATASET_RATIO,
        )
    else:
        raise ValueError('Unknown training sampler: {}'.format(sampler_name))

    return {
        'dataset': dataset_dicts,
        'sampler': sampler,
        'mapper': mapper,
        'total_batch_size': cfg.SOLVER.IMS_PER_BATCH,
        'num_workers': cfg.DATALOADER.NUM_WORKERS,
        'dataset_bs': cfg.DATALOADER.DATASET_BS,
        'num_datasets': len(cfg.DATASETS.TRAIN)
    }


@configurable(from_config=_custom_train_loader_from_config)
def build_custom_train_loader(dataset,
                              *,
                              mapper,
                              sampler,
                              total_batch_size=16,
                              num_workers=0,
                              num_datasets=1,
                              dataset_bs=1):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)

    return build_dataset_batch_data_loader(
        dataset_bs,
        dataset,
        sampler,
        total_batch_size,
        num_datasets=num_datasets,
        num_workers=num_workers,
    )


def build_dataset_batch_data_loader(dataset_bs,
                                    dataset,
                                    sampler,
                                    total_batch_size,
                                    num_datasets,
                                    num_workers=0):
    world_size = get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), 'Total batch size ({}) must be divisible by the number of gpus ({}).' \
        .format(total_batch_size, world_size)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(
            0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )

    if num_datasets > 1:
        return MultiDatasets(data_loader, dataset_bs, num_datasets)
    else:
        return SingleDataset(data_loader, dataset_bs)


def get_detection_dataset_dicts_with_source(dataset_names,
                                            filter_empty=True,
                                            min_keypoints=0,
                                            proposal_files=None):
    assert len(dataset_names)
    dataset_dicts = [
        DatasetCatalog.get(dataset_name) for dataset_name in dataset_names
    ]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    for source_id, (dataset_name, dicts) in \
            enumerate(zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        for d in dicts:
            d['dataset_source'] = source_id

        if 'annotations' in dicts[0]:
            try:
                class_names = MetadataCatalog.get(dataset_name).thing_classes
                check_metadata_consistency('thing_classes', dataset_name)
                print_instances_class_histogram(dicts, class_names)
            except AttributeError:
                pass

    assert proposal_files is None

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = 'annotations' in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(
            dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(
            dataset_dicts, min_keypoints)

    return dataset_dicts


class MultiDatasetSampler(Sampler):

    def __init__(
        self,
        dataset_dicts,
        dataset_ratio,
        seed: Optional[int] = None,
    ):
        sizes = [0 for _ in range(len(dataset_ratio))]
        for d in dataset_dicts:
            sizes[d['dataset_source']] += 1
        print('dataset sizes', sizes)
        self.sizes = sizes
        assert len(dataset_ratio) == len(sizes), \
            'length of dataset ' \
            'ratio {} should be equal to number if dataset {}'.format(
                len(dataset_ratio), len(sizes)
            )
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self.dataset_ids = torch.tensor(
            [d['dataset_source'] for d in dataset_dicts], dtype=torch.long)
        self.dataset_ratio = dataset_ratio

        dataset_weight = [
            torch.ones(s) * max(sizes) / s * r / sum(dataset_ratio)
            for i, (r, s) in enumerate(zip(dataset_ratio, sizes))
        ]
        dataset_weight = torch.cat(dataset_weight)

        self.weights = dataset_weight
        self.sample_epoch_size = len(self.weights)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None,
                                    self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if len(self.dataset_ratio) > 1:
                # multiple datasets
                ids = torch.multinomial(
                    self.weights,
                    self.sample_epoch_size,
                    generator=g,
                    replacement=True)
                # nums = [(self.dataset_ids[ids] == i).sum().int().item()
                #         for i in range(len(self.sizes))]
                yield from ids
            else:
                # single dataset
                yield from torch.randperm(self.sizes[0], generator=g).tolist()


class SingleDataset(torch.utils.data.IterableDataset):

    def __init__(self, dataset, batch_sizes):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(2)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d['width'], d['height']
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_sizes:
                yield bucket[:]
                del bucket[:]


class MultiDatasets(torch.utils.data.IterableDataset):

    def __init__(self, dataset, batch_sizes, num_datasets):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(2 * num_datasets)]
        self.iter_idx = 0
        self.num_datasets = num_datasets

    def __iter__(self):
        for d in self.dataset:
            w, h = d['width'], d['height']
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            if len(bucket) < self.batch_sizes:
                bucket.append(d)
            selected_dataset = self.iter_idx % self.num_datasets
            if len(bucket) == self.batch_sizes \
                    and selected_dataset == d['dataset_source']:
                self.iter_idx += 1
                yield bucket[:]
                del bucket[:]
