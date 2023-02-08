# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from functools import partial
from multiprocessing import Manager, cpu_count

import numpy as np
from mmengine import Config, DictAction, track_parallel_progress
from mmengine.registry import init_default_scope

from mmaction.registry import DATASETS, TRANSFORMS


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 check datasets')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--output-file',
        default='invalid-video.txt',
        help='Output file path which keeps corrupted/missing video file paths')
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split')
    parser.add_argument(
        '--decoder',
        default='decord',
        choices=['decord', 'opencv', 'pyav'],
        help='Video decoder type, should be one of [decord, opencv, pyav]')
    parser.add_argument(
        '--nproc',
        type=int,
        default=(cpu_count() - 1 or 1),
        help='Number of processes to check videos')
    parser.add_argument(
        '--remove-corrupted-videos',
        action='store_true',
        help='Whether to delete all corrupted videos')
    args = parser.parse_args()

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


@TRANSFORMS.register_module()
class RandomSampleFrames:

    def __call__(self, results):
        """Select frames to verify.

        Select the first, last and three random frames, Required key is
        "total_frames", added or modified key is "frame_inds".
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        assert results['total_frames'] > 0

        # first and last frames
        results['frame_inds'] = np.array([0, results['total_frames'] - 1])

        # choose 3 random frames
        if results['total_frames'] > 2:
            results['frame_inds'] = np.concatenate([
                results['frame_inds'],
                np.random.randint(1, results['total_frames'] - 1, 3)
            ])

        return results


def _do_check_videos(lock, pipeline, output_file, data_info):
    try:
        pipeline(data_info)
    except:  # noqa
        # save invalid video path to output file
        lock.acquire()
        with open(output_file, 'a') as f:
            f.write(data_info['filename'] + '\n')
        lock.release()


if __name__ == '__main__':
    args = parse_args()

    decoder_to_pipeline_prefix = dict(
        decord='Decord', opencv='OpenCV', pyav='PyAV')

    # read config file
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmaction'))

    # build dataset
    dataset_cfg = cfg.get(f'{args.split}_dataloader').dataset
    dataset_type = dataset_cfg.type
    assert dataset_type == 'VideoDataset'
    dataset_cfg.pipeline = [
        dict(type=decoder_to_pipeline_prefix[args.decoder] + 'Init'),
        dict(type='RandomSampleFrames'),
        dict(type=decoder_to_pipeline_prefix[args.decoder] + 'Decode')
    ]

    dataset = DATASETS.build(dataset_cfg)
    dataset_cfg.pop('type')
    pipeline = dataset.pipeline

    # prepare for checking
    if os.path.exists(args.output_file):
        # remove existing output file
        os.remove(args.output_file)

    lock = Manager().Lock()
    worker_fn = partial(_do_check_videos, lock, pipeline, args.output_file)
    # avoid copy dataset for multiprocess
    data_info_list = [
        dataset.get_data_info(idx) for idx in range(len(dataset))
    ]

    # start checking
    track_parallel_progress(worker_fn, data_info_list, nproc=args.nproc)

    if os.path.exists(args.output_file):
        num_lines = sum(1 for _ in open(args.output_file))
        print(f'Checked {len(dataset)} videos, '
              f'{num_lines} are corrupted/missing.')
        if args.remove_corrupted_videos:
            print('Start deleting corrupted videos')
            cnt = 0
            with open(args.output_file, 'r') as f:
                for line in f:
                    if os.path.exists(line.strip()):
                        os.remove(line.strip())
                        cnt += 1
            print(f'Deleted {cnt} corrupted videos.')
    else:
        print(f'Checked {len(dataset)} videos, none are corrupted/missing')
