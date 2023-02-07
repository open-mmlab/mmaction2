# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmaction.apis.inferencers import MMAction2Inferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input video file or rawframes folder path.')
    parser.add_argument(
        '--vid-out-dir',
        type=str,
        default='',
        help='Output directory of videos.')
    parser.add_argument(
        '--rec',
        type=str,
        default=None,
        help='Pretrained action recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected recog model. '
        'If it is not specified and "rec" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--label-file', type=str, default=None, help='label file for dataset.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the video in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--pred-out-file',
        type=str,
        default='',
        help='File to save the inference results.')

    call_args = vars(parser.parse_args())

    init_kws = ['rec', 'rec_weights', 'device', 'label_file']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    mmaction2 = MMAction2Inferencer(**init_args)
    mmaction2(**call_args)


if __name__ == '__main__':
    main()
