# Copyright (c) OpenMMLab. All rights reserved.
# This script serves the sole purpose of converting skeleton-based graph
# in MMAction2 to ONNX files. Please note that attempting to convert other
# models using this script may not yield successful results.
import argparse

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import LabelData

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--num_frames', type=int, default=150, help='number of input frames.')
    parser.add_argument(
        '--num_person', type=int, default=2, help='number of maximum person.')
    parser.add_argument(
        '--num_joints',
        type=int,
        default=0,
        help='number of joints. If not given, will use default settings from'
        'the config file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--output_file',
        type=str,
        default='stgcn.onnx',
        help='file name of the output onnx file')
    args = parser.parse_args()
    return args


class AvgPool2d(nn.Module):

    def forward(self, x):
        return x.mean(dim=(-1, -2), keepdims=True)


class MaxPool2d(nn.Module):

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.max(dim=-2, keepdim=True)[0]
        return x


class GCNNet(nn.Module):

    def __init__(self, base_model):
        super(GCNNet, self).__init__()
        self.backbone = base_model.backbone
        self.head = base_model.cls_head

        if hasattr(self.head, 'pool'):
            pool = self.head.pool
            if isinstance(pool, nn.AdaptiveAvgPool2d):
                assert pool.output_size == 1
                self.head.pool = AvgPool2d()
            elif isinstance(pool, nn.AdaptiveMaxPool2d):
                assert pool.output_size == 1
                self.head.pool = MaxPool2d()

    def forward(self, input_tensor):
        feat = self.backbone(input_tensor)
        cls_score = self.head(feat)
        return cls_score


def softmax(x):
    x = np.exp(x - x.max())
    return x / x.sum()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    init_default_scope(config.get('default_scope', 'mmaction'))

    if config.model.type != 'RecognizerGCN':
        print(
            'This script serves the sole purpose of converting skeleton-based '
            'graph in MMAction2 to ONNX files. Please note that attempting to '
            'convert other models using this script may not yield successful '
            'results.\n\n')

    base_model = MODELS.build(config.model)
    load_checkpoint(base_model, args.checkpoint, map_location='cpu')
    base_model.to(args.device)

    lookup = {'openpose': 18, 'nturgb+d': 25, 'coco': 17}

    num_joints = args.num_joints
    num_person = args.num_person
    num_frames = args.num_frames
    if num_joints == 0:
        layout = config.model.backbone.graph_cfg.layout
        if layout not in lookup:
            raise KeyError(
                '`layout` not supported, please specify `num_joints`')
        num_joints = lookup[layout]

    input_tensor = torch.randn(1, num_person, num_frames, num_joints, 3)
    input_tensor = input_tensor.clamp(-3, 3).to(args.device)

    base_model.eval()

    data_sample = ActionDataSample()
    data_sample.pred_scores = LabelData()
    data_sample.pred_labels = LabelData()
    base_output = base_model(
        input_tensor.unsqueeze(0), data_samples=[data_sample],
        mode='predict')[0]
    base_output = base_output.pred_score.detach().cpu().numpy()

    model = GCNNet(base_model).to(args.device)
    model.eval()

    torch.onnx.export(
        model, (input_tensor),
        args.output_file,
        input_names=['input_tensor'],
        output_names=['cls_score'],
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=12,
        dynamic_axes={
            'input_tensor': {
                0: 'batch_size',
                1: 'num_person',
                2: 'num_frames'
            },
            'cls_score': {
                0: 'batch_size'
            }
        })

    print(f'Successfully export the onnx file to {args.output_file}')

    # Test exported file
    session = onnxruntime.InferenceSession(args.output_file)
    input_feed = {'input_tensor': input_tensor.cpu().data.numpy()}
    outputs = session.run(['cls_score'], input_feed=input_feed)
    output = softmax(outputs[0][0])

    diff = abs(base_output - output).max()
    if diff < 1e-5:
        print('The output difference is smaller than 1e-5.')


if __name__ == '__main__':
    main()
