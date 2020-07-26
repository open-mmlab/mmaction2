import argparse
import os.path as osp
import pickle

import torch
from tqdm import tqdm

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    parser.add_argument('--data-prefix', default='', help='dataset prefix')
    parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument(
        '--data-list',
        help='video list of the \
        dataset, the format should be `frame_dir num_frames output_file`')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=16,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--clip-len', type=int, default=1, help='clip length')
    parser.add_argument('--modality', default='RGB')
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.is_rgb = args.modality == 'RGB'
    args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
    rgb_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'image_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)

    data_pipeline = [
        dict(
            type='UntrimSampleFrames',
            clip_len=args.clip_len,
            frame_interval=args.frame_interval),
        dict(type='FrameSelector'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=256),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format=args.input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    model = dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNet',
            depth=50,
            in_channels=args.in_channels,
            norm_eval=False),
        cls_head=dict(
            type='TSNHead',
            num_classes=200,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1)))
    model = build_model(model, test_cfg=dict(average_clips=None))
    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    data = open(args.data_list).readlines()
    data = [x.strip() for x in data]
    for item in tqdm(data):
        item = item.split()
        frame_dir, length, output_file = item
        frame_dir = osp.join(args.data_prefix, frame_dir)
        output_file = osp.join(args.output_prefix, output_file)
        assert output_file.endswith('.pkl')
        length = int(length)
        # just use pseudo label here
        tmpl = dict(
            frame_dir=frame_dir,
            total_frames=length,
            filename_tmpl=args.f_tmpl,
            modality=args.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs']
        shape = imgs.shape
        imgs = imgs.reshape((
            shape[0],
            1,
        ) + shape[1:])
        imgs = imgs.cuda()
        with torch.no_grad():
            feat = model.forward(imgs, return_loss=False)
        with open(output_file, 'wb') as fout:
            pickle.dump(feat, fout)


if __name__ == '__main__':
    main()
