# Copyright (c) OpenMMLab. All rights reserved.
# This script serves the sole purpose of converting spatial-temporal detection
# models supported in MMAction2 to ONNX files. Please note that attempting to
# convert other models using this script may not yield successful results.
import argparse

import onnxruntime
import torch
import torch.nn as nn
from mmdet.structures.bbox import bbox2roi
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmaction.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--num_frames', type=int, default=8, help='number of input frames.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 455],
        help='input image size')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--output_file',
        type=str,
        default='stdet.onnx',
        help='file name of the output onnx file')
    args = parser.parse_args()
    return args


class SpatialMaxPool3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        return x.max(dim=-2, keepdim=True)[0]


class SpatialAvgPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=(-1, -2), keepdims=True)


class TemporalMaxPool3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(dim=-3, keepdim=True)[0]


class TemporalAvgPool3d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=-3, keepdim=True)


class GlobalPool2d(nn.Module):

    def __init__(self, pool_size, output_size, later_max=True):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size)
        self.max = later_max
        self.output_size = output_size

    def forward(self, x):
        x = self.pool(x)
        if self.max:
            x = x.max(dim=-1, keepdim=True)[0]
            x = x.max(dim=-2, keepdim=True)[0]
        else:
            x = x.mean(dim=(-1, -2), keepdims=True)
        x = x.expand(-1, -1, self.output_size, self.output_size)
        return x


class STDet(nn.Module):

    def __init__(self, base_model, input_tensor):
        super(STDet, self).__init__()
        self.backbone = base_model.backbone
        self.bbox_roi_extractor = base_model.roi_head.bbox_roi_extractor
        self.bbox_head = base_model.roi_head.bbox_head

        output_size = self.bbox_roi_extractor.global_pool.output_size
        pool_size = min(input_tensor.shape[-2:]) // 16 // output_size

        if isinstance(self.bbox_head.temporal_pool, nn.AdaptiveAvgPool3d):
            self.bbox_head.temporal_pool = TemporalAvgPool3d()
        else:
            self.bbox_head.temporal_pool = TemporalMaxPool3d()
        if isinstance(self.bbox_head.spatial_pool, nn.AdaptiveAvgPool3d):
            self.bbox_head.spatial_pool = SpatialAvgPool()
            self.bbox_roi_extractor.global_pool = GlobalPool2d(
                pool_size, output_size, later_max=False)
        else:
            self.bbox_head.spatial_pool = SpatialMaxPool3d()
            self.bbox_roi_extractor.global_pool = GlobalPool2d(
                pool_size, output_size, later_max=True)

    def forward(self, input_tensor, rois):
        feat = self.backbone(input_tensor)
        bbox_feats, _ = self.bbox_roi_extractor(feat, rois)
        cls_score = self.bbox_head(bbox_feats)
        return cls_score


def main():
    args = parse_args()
    config = Config.fromfile(args.config)

    if config.model.type != 'FastRCNN':
        print('This script serves the sole purpose of converting spatial '
              'temporal detection models in MMAction2 to ONNX files. Please '
              'note that attempting to convert other models using this script '
              'may not yield successful results.\n\n')

    init_default_scope(config.get('default_scope', 'mmaction'))

    base_model = MODELS.build(config.model)
    load_checkpoint(base_model, args.checkpoint, map_location='cpu')
    base_model.to(args.device)

    if len(args.shape) == 1:
        input_shape = (args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    input_tensor = torch.randn(1, 3, args.num_frames, *input_shape)
    input_tensor = input_tensor.clamp(-3, 3).to(args.device)
    proposal = torch.Tensor([[22., 59., 67., 157.], [186., 73., 217., 159.],
                             [407., 95., 431., 168.]])

    rois = bbox2roi([proposal]).to(args.device)

    model = STDet(base_model, input_tensor).to(args.device)
    model.eval()
    cls_score = model(input_tensor, rois)
    print(f'Model output shape: {cls_score.shape}')

    torch.onnx.export(
        model, (input_tensor, rois),
        args.output_file,
        input_names=['input_tensor', 'rois'],
        output_names=['cls_score'],
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=11,
        dynamic_axes={
            'input_tensor': {
                0: 'batch_size',
                3: 'height',
                4: 'width'
            },
            'rois': {
                0: 'total_num_bbox_for_the_batch'
            },
            'cls_score': {
                0: 'total_num_bbox_for_the_batch'
            }
        })

    print(f'Successfully export the onnx file to {args.output_file}')

    # Test exported file
    session = onnxruntime.InferenceSession(args.output_file)
    input_feed = {
        'input_tensor': input_tensor.cpu().data.numpy(),
        'rois': rois.cpu().data.numpy()
    }
    outputs = session.run(['cls_score'], input_feed=input_feed)
    outputs = outputs[0]
    diff = abs(cls_score.cpu().data.numpy() - outputs).max()
    if diff < 1e-5:
        print('The output difference is smaller than 1e-5.')


if __name__ == '__main__':
    main()
