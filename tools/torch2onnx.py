import argparse
import os.path as osp
import warnings

import mmcv
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmaction.models import build_model

try:
    import onnx
except ImportError:
    warnings.warn('Please install onnx to support onnx exporting.')


class RecognizerWrapper(nn.Module):
    """Wrapper that only inferences the part in computation graph."""

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, x):
        return self.recognizer.forward_dummy(x)


class LocalizerWrapper(nn.Module):
    """Wrapper that only inferences the part in computation graph."""

    def __init__(self, localizer):
        super().__init__()
        self.localizer = localizer

    def forward(self, x):
        return self.localizer._forward(x)


def _get_cfg(config_path):
    """Grab configs necessary to create a model."""
    if not osp.exists(config_path):
        raise FileNotFoundError('Cannot find config path')
    config = mmcv.Config.fromfile(config_path)
    return config.model, config.data.test.pipeline, config.test_cfg


def torch2onnx(input, model):
    exported_name = osp.basename(args.checkpoint).replace('.pth', '.onnx')
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(
        model,
        input,
        exported_name,
        verbose=False,
        # Using a higher version of onnx opset
        opset_version=11,
        input_names=input_names,
        output_names=output_names)
    model = onnx.load(exported_name)
    onnx.checker.check_model(model)


def parse_args():
    parser = argparse.ArgumentParser(description='Export a model to onnx')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        default=False,
        help='Determine whether the model is a localizer')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs='+',
        default=None,
        help='Input dimension, mandatory for localizers')
    args = parser.parse_args()
    args.input_size = tuple(args.input_size) if args.input_size else None
    return args


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    checkpoint_path = args.checkpoint

    model_cfg, test_pipeline, test_cfg = _get_cfg(config_path)

    model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg).cuda()
    if not args.is_localizer:
        try:
            dummy_input = torch.randn(args.input_size).cuda()
        except TypeError:
            for trans in test_pipeline:
                if trans['type'] == 'SampleFrames':
                    t = trans['clip_len']
                    n = trans['num_clips']
                elif trans['type'] == 'Resize':
                    if isinstance(trans['scale'], int):
                        s = trans['scale']
                    elif isinstance(trans['scale'], tuple):
                        s = max(trans['scale'])
            # #crop x (#batch * #clip) x #channel x clip_len x height x width
            dummy_input = torch.randn(1, 1 * n, 3, t, s, s).cuda()
        # squeeze the t-dimension for 2d model
        dummy_input = dummy_input.squeeze(3)
        wrapped_model = RecognizerWrapper(model)
    else:
        try:
            # #batch x #channel x length
            dummy_input = torch.randn(args.input_size).cuda()
        except TypeError as e:
            print(f'{e}\nplease specify the input size for localizer.')
            exit()
        wrapped_model = LocalizerWrapper(model)
    load_checkpoint(
        getattr(wrapped_model,
                'recognizer' if not args.is_localizer else 'localizer'),
        checkpoint_path)
    torch2onnx(dummy_input, wrapped_model)
