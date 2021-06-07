import argparse
import os
import warnings

from mmcv import Config, DictAction
from tqdm import tqdm

from mmaction.datasets import build_dataset


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


if __name__ == '__main__':
    args = parse_args()

    assert args.split in ['train', 'val', 'test']

    decoder_to_pipeline = dict(
        decord='DecordInit',
        opencv='OpenCVInit',
        pyav='PyAVInit',
    )
    assert args.decoder in decoder_to_pipeline

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    dataset_type = cfg.data[args.split].type
    assert dataset_type == 'VideoDataset'

    # Only video decoder is needed for the data pipeline
    cfg.data[args.split].pipeline = [
        dict(type=decoder_to_pipeline[args.decoder])
    ]
    dataset = build_dataset(cfg.data[args.split],
                            dict(test_mode=(args.split != 'train')))

    writer = open(args.output_file, 'w')
    cnt = 0
    for i in tqdm(range(len(dataset))):
        try:
            dataset[i]
        except:  # noqa
            # save invalid video path to output file
            writer.write(dataset.video_infos[i]['filename'] + '\n')
            cnt += 1

    print(f'Checked {len(dataset)} videos, {cnt} is/are corrupted/missing.')
    writer.close()

    if args.remove_corrupted_videos:
        print('Start deleting corrupted videos')
        cnt = 0
        with open(args.output_file, 'r') as f:
            for line in f:
                if os.path.exists(line.strip()):
                    os.remove(line.strip())
                    cnt += 1
        print(f'Delete {cnt} corrupted videos.')
