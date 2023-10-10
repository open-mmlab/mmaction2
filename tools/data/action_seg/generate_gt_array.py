# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

dataset_names = ['50salads', 'breakfast', 'gtea']


def get_class2id_map(dataset: str,
                     dataset_dir: str = './dataset') -> Dict[str, int]:
    """
    Args:
        dataset: 50salads, gtea, breakfast
        dataset_dir: the path to the dataset directory
    """

    assert (dataset in dataset_names
            ), 'You have to choose 50salads, gtea or breakfast as dataset.'

    with open(
            os.path.join(dataset_dir, '{}/mapping.txt'.format(dataset)),
            'r') as f:
        actions = f.read().split('\n')[:-1]

    class2id_map = dict()
    for a in actions:
        class2id_map[a.split()[1]] = int(a.split()[0])

    return class2id_map


def get_id2class_map(dataset: str,
                     dataset_dir: str = './dataset') -> Dict[int, str]:
    class2id_map = get_class2id_map(dataset, dataset_dir)

    return {val: key for key, val in class2id_map.items()}


def get_n_classes(dataset: str, dataset_dir: str = './dataset') -> int:
    return len(get_class2id_map(dataset, dataset_dir))


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line interface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description='convert ground truth txt files to numpy array')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset',
        help='path to a dataset directory (default: ./dataset)',
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    datasets = ['50salads', 'gtea', 'breakfast']

    for dataset in datasets:
        # make directory for saving ground truth numpy arrays
        save_dir = os.path.join(args.dataset_dir, dataset, 'gt_arr')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # class to index mapping
        class2id_map = get_class2id_map(dataset, dataset_dir=args.dataset_dir)

        gt_dir = os.path.join(args.dataset_dir, dataset, 'groundTruth')
        gt_paths = glob.glob(os.path.join(gt_dir, '*.txt'))

        for gt_path in gt_paths:
            # the name of ground truth text file
            gt_name = os.path.relpath(gt_path, gt_dir)

            with open(gt_path, 'r') as f:
                gt = f.read().split('\n')[:-1]

            gt_array = np.zeros(len(gt))
            for i in range(len(gt)):
                gt_array[i] = class2id_map[gt[i]]

            # save array
            np.save(os.path.join(save_dir, gt_name[:-4] + '.npy'), gt_array)

    print('Done')


if __name__ == '__main__':
    main()
