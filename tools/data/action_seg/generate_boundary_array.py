# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os

import numpy as np


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line interface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description='generate ground truth arrays for boundary regression.')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='path to a dataset directory',
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    datasets = ['50salads', 'gtea', 'breakfast']

    for dataset in datasets:
        # make directory for saving ground truth numpy arrays
        save_dir = os.path.join(args.dataset_dir, dataset, 'gt_boundary_arr')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        gt_dir = os.path.join(args.dataset_dir, dataset, 'groundTruth')
        gt_paths = glob.glob(os.path.join(gt_dir, '*.txt'))

        for gt_path in gt_paths:
            # the name of ground truth text file
            gt_name = os.path.relpath(gt_path, gt_dir)

            with open(gt_path, 'r') as f:
                gt = f.read().split('\n')[:-1]

            # define the frame where new action starts as boundary frame
            boundary = np.zeros(len(gt))
            last = gt[0]
            boundary[0] = 1
            for i in range(1, len(gt)):
                if last != gt[i]:
                    boundary[i] = 1
                    last = gt[i]

            # save array
            np.save(os.path.join(save_dir, gt_name[:-4] + '.npy'), boundary)

    print('Done')


if __name__ == '__main__':
    main()
