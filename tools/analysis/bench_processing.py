"""This file is for benchmark dataloading process. The command line to run this
file is:

$ python -m cProfile -o program.prof tools/analysis/bench_processing.py
configs/task/method/[config filename]

It use cProfile to record cpu running time and output to program.prof
To visualize cProfile output program.prof, use Snakeviz and run:
$ snakeviz program.prof
"""
import argparse
import os

import mmcv
from mmcv import Config

from mmaction import __version__
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import get_root_logger


def main():
    parser = argparse.ArgumentParser(description='Benchmark dataloading')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init logger before other steps
    logger = get_root_logger()
    logger.info(f'MMAction2 Version: {__version__}')
    logger.info(f'Config: {cfg.text}')

    # create bench data list
    ann_file_bench = 'benchlist.txt'
    if not os.path.exists(ann_file_bench):
        with open(cfg.ann_file_train) as f:
            lines = f.readlines()[:256]
            with open(ann_file_bench, 'w') as f1:
                f1.writelines(lines)
    cfg.data.train.ann_file = ann_file_bench

    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=cfg.data.videos_per_gpu,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False)

    # Start progress bar after first 5 batches
    prog_bar = mmcv.ProgressBar(
        len(dataset) - 5 * cfg.data.videos_per_gpu, start=False)
    for i, data in enumerate(data_loader):
        if i == 5:
            prog_bar.start()
        for _ in data['imgs']:
            if i < 5:
                continue
            prog_bar.update()


if __name__ == '__main__':
    main()
