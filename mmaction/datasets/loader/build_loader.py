import platform
from functools import partial

from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader, DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     pin_memory=True,
                     **kwargs):
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        **kwargs)

    return data_loader
