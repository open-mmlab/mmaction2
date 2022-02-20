# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from mmcv import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('audio', help='audio file')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    model = init_recognizer(cfg, args.checkpoint, device=device)

    if not args.audio.endswith('.npy'):
        raise NotImplementedError('Demo works on extracted audio features')
    results = inference_recognizer(model, args.audio)

    labels = open(args.label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in results]

    print('Scores:')
    for result in results:
        print(f'{result[0]}: ', result[1])


if __name__ == '__main__':
    main()
