import argparse
import os
import warnings
from functools import partial
from multiprocessing import Manager, Pool

import numpy as np
from mmcv import Config, DictAction
from tqdm import tqdm

from mmaction.datasets import PIPELINES, build_dataset


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
        help='Dataset split, should be one of [train, val, test]')
    parser.add_argument(
        '--decoder',
        default='decord',
        help='Video decoder type, should be one of [decord, opencv, pyav]')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=5,
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


@PIPELINES.register_module()
class RandomSampleFrames:

    def __call__(self, results):
        """Select frames to verify.

        Required key is "total_frames", added or modified key is "frame_inds".
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        assert results['total_frames'] > 0

        # first and last elements
        results['frame_inds'] = np.array([0, results['total_frames'] - 1])

        # choose 3 random frames
        if results['total_frames'] > 2:
            results['frame_inds'] = np.concatenate([
                results['frame_inds'],
                np.random.randint(1, results['total_frames'] - 1, 3)
            ])

        return results


def _do_check_videos(lock, dataset, output_file, idx):
    try:
        dataset[idx]
    except:  # noqa
        # save invalid video path to output file
        lock.acquire()
        with open(output_file, 'a') as f:
            f.write(dataset.video_infos[idx]['filename'] + '\n')
        lock.release()


if __name__ == '__main__':
    args = parse_args()

    assert args.split in ['train', 'val', 'test']

    decoder_to_pipeline_prefix = dict(
        decord='Decord',
        opencv='OpenCV',
        pyav='PyAV',
    )
    assert args.decoder in decoder_to_pipeline_prefix

    # read config file
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # build dataset
    dataset_type = cfg.data[args.split].type
    assert dataset_type == 'VideoDataset'
    cfg.data[args.split].pipeline = [
        dict(type=decoder_to_pipeline_prefix[args.decoder] + 'Init'),
        dict(type='RandomSampleFrames'),
        dict(type=decoder_to_pipeline_prefix[args.decoder] + 'Decode')
    ]
    dataset = build_dataset(cfg.data[args.split],
                            dict(test_mode=(args.split != 'train')))

    # prepare for checking
    if os.path.exists(args.output_file):
        # remove exsiting output file
        os.remove(args.output_file)
    pool = Pool(args.num_processes)
    lock = Manager().Lock()
    worker_fn = partial(_do_check_videos, lock, dataset, args.output_file)
    ids = range(len(dataset))

    # start checking
    for _ in tqdm(pool.imap_unordered(worker_fn, ids), total=len(ids)):
        pass
    pool.join()

    # print results and release resources
    pool.close()
    with open(args.output_file, 'r') as f:
        print(f'Checked {len(dataset)} videos, '
              f'{len(f)} is/are corrupted/missing.')

    if args.remove_corrupted_videos:
        print('Start deleting corrupted videos')
        cnt = 0
        with open(args.output_file, 'r') as f:
            for line in f:
                if os.path.exists(line.strip()):
                    os.remove(line.strip())
                    cnt += 1
        print(f'Delete {cnt} corrupted videos.')
