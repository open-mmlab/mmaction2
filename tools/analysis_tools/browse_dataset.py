# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import sys

import mmcv
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose

from mmaction.registry import DATASETS, VISUALIZERS
from mmaction.utils import register_all_modules
from mmaction.visualization import ActionVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--show-frames',
        default=False,
        action='store_true',
        help='Whether to display the frames of the video. Defaults to False,'
        'Please make sure you have the display interface')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-number',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--mode',
        default='transformed',
        type=str,
        choices=['original', 'transformed'],
        help='display mode; display original videos or transformed videos.'
        '"original" means show videos load from disk;'
        '"transformed" means to show videos after transformed; '
        'Defaults to "transformed".')
    parser.add_argument(
        '--rescale-factor',
        type=float,
        help='video rescale factor, which is useful if the output is too '
        'large or too small.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


class InspectCompose(Compose):
    """Compose multiple transforms sequentially.

    And record "imgs" field of all results in one list.
    """

    def __init__(self, transforms, intermediate_imgs):
        super().__init__(transforms=transforms)
        self.intermediate_imgs = intermediate_imgs

    def __call__(self, data):

        for idx, t in enumerate(self.transforms):
            data = t(data)
            if data is None:
                return None
            if 'imgs' in data:
                name = t.__class__.__name__
                imgs = data['imgs'].copy()
                if name != 'FormatShape':
                    self.intermediate_imgs.append({'name': name, 'imgs': imgs})
        return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmaction2 into the registries
    register_all_modules()

    dataset_cfg = cfg.get(args.phase + '_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)

    intermediate_imgs = []
    dataset.pipeline = InspectCompose(dataset.pipeline.transforms,
                                      intermediate_imgs)

    # init visualizer
    default_visualizer = {
        'type': 'ActionVisualizer',
        'name': 'dataset_browser',
        'save_dir': 'temp_browse_dataset'
    }
    visualizer = cfg.get('visualizer', default_visualizer)
    visualizer: ActionVisualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    # init visualization video number
    display_number = min(args.show_number, len(dataset))
    progress_bar = mmcv.ProgressBar(display_number)

    for i, item in zip(range(display_number), dataset):
        if args.mode == 'original':
            video = intermediate_imgs[0]['imgs']
        elif args.mode == 'transformed':
            video = intermediate_imgs[-1]['imgs']
        else:
            raise NameError('Currently %s mode is not supported!' % args.mode)
        intermediate_imgs.clear()

        data_sample = item['data_sample'].numpy()

        file_id = f'video_{i}'
        out_folder = osp.join(args.output_dir,
                              file_id) if args.output_dir is not None else None

        visualizer.add_datasample(
            file_id,
            video,
            data_sample,
            show_frames=args.show_frames,
            out_folder=out_folder)
        progress_bar.update()


if __name__ == '__main__':
    main()
