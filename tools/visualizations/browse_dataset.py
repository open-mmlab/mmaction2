# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
import warnings
from copy import deepcopy

import cv2
import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmengine.visualization import Visualizer

from mmaction.registry import DATASETS
from mmaction.visualization import ActionVisualizer
from mmaction.visualization.action_visualizer import _get_adaptive_scale


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        'output_dir', default=None, type=str, help='output directory')
    parser.add_argument('--label', default=None, type=str, help='label file')
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--fps',
        default=5,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'concat', 'pipeline'],
        help='display mode; display original pictures or transformed pictures'
        ' or comparison pictures. "original" means show images load from disk'
        '; "transformed" means to show images after transformed; "concat" '
        'means show images stitched by "original" and "output" images. '
        '"pipeline" means show all the intermediate images. '
        'Defaults to "transformed".')
    parser.add_argument(
        '--rescale-factor',
        '-r',
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


def make_grid(videos, names, rescale_factor=None):
    """Concat list of pictures into a single big picture, align height here."""
    vis = Visualizer()

    ori_shapes = [vid[0].shape[:2] for vid in videos]
    if rescale_factor is not None:
        videos = [[mmcv.imrescale(img, rescale_factor) for img in video]
                  for video in videos]

    max_height = int(max(vid[0].shape[0] for vid in videos) * 1.4)
    min_width = min(vid[0].shape[1] for vid in videos)
    horizontal_gap = min_width // 10
    img_scale = _get_adaptive_scale((max_height, min_width))

    texts = []
    text_positions = []
    start_x = 0
    for i, vid in enumerate(videos):
        for j, img in enumerate(vid):
            pad_height = (max_height - img.shape[0]) // 2
            pad_width = horizontal_gap // 2
            # make border
            videos[i][j] = cv2.copyMakeBorder(
                img,
                pad_height,
                max_height - img.shape[0] - pad_height +
                int(img_scale * 30 * 2),
                pad_width,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255))

        texts.append(f'{names[i]}\n{ori_shapes[i]}')
        text_positions.append(
            [start_x + img.shape[1] // 2 + pad_width, max_height])
        start_x += img.shape[1] + horizontal_gap

    out_frames = []
    for i in range(len(videos[0])):
        imgs = [vid[i] for vid in videos]
        display_img = np.concatenate(imgs, axis=1)
        vis.set_image(display_img)
        img_scale = _get_adaptive_scale(display_img.shape[:2])
        vis.draw_texts(
            texts,
            positions=np.array(text_positions),
            font_sizes=img_scale * 7,
            colors='black',
            horizontal_alignments='center',
            font_families='monospace')
        out_frames.append(vis.get_image())
    return out_frames


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
                imgs = deepcopy(data['imgs'])
                if name == 'FormatShape':
                    continue
                if name == 'ThreeCrop':
                    n_crops = 3
                    clip_len = len(imgs) // n_crops
                    crop_imgs = [
                        imgs[idx * clip_len:(idx + 1) * clip_len]
                        for idx in range(n_crops)
                    ]
                    imgs = np.concatenate(crop_imgs, axis=1)
                    imgs = [img for img in imgs]
                if name == 'TenCrop':
                    warnings.warn(
                        'TenCrop is not supported, only show one crop')
                self.intermediate_imgs.append({'name': name, 'imgs': imgs})
        return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    init_default_scope(cfg.get('default_scope', 'mmaction'))

    dataset_cfg = cfg.get(args.phase + '_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)

    intermediate_imgs = []
    dataset.pipeline = InspectCompose(dataset.pipeline.transforms,
                                      intermediate_imgs)

    # init visualizer
    vis_backends = [dict(
        type='LocalVisBackend',
        save_dir=args.output_dir,
    )]
    visualizer = ActionVisualizer(
        vis_backends=vis_backends, save_dir='place_holder')

    if args.label:
        labels = open(args.label).readlines()
        labels = [x.strip() for x in labels]
        visualizer.dataset_meta = dict(classes=labels)

    # init visualization video number
    display_number = min(args.show_number, len(dataset))
    progress_bar = ProgressBar(display_number)

    for i, item in zip(range(display_number), dataset):
        rescale_factor = args.rescale_factor
        if args.mode == 'original':
            video = intermediate_imgs[0]['imgs']
        elif args.mode == 'transformed':
            video = intermediate_imgs[-1]['imgs']
        elif args.mode == 'concat':
            ori_video = intermediate_imgs[0]['imgs']
            trans_video = intermediate_imgs[-1]['imgs']
            video = make_grid([ori_video, trans_video],
                              ['original', 'transformed'], rescale_factor)
            rescale_factor = None
        else:
            video = make_grid([result['imgs'] for result in intermediate_imgs],
                              [result['name'] for result in intermediate_imgs],
                              rescale_factor)
            rescale_factor = None

        intermediate_imgs.clear()

        data_sample = item['data_samples'].numpy()

        file_id = f'video_{i}'
        video = [x[..., ::-1] for x in video]
        visualizer.add_datasample(
            file_id, video, data_sample, fps=args.fps, out_type='video')
        progress_bar.update()


if __name__ == '__main__':
    main()
