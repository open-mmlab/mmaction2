import os.path as osp
import sys

import mmcv
import onnx
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_model


def _get_recognizer_cfg(config_path):
    """Grab configs necessary to create a recognizer."""
    if not osp.exists(config_path):
        raise FileNotFoundError('Cannot find config path')
    config = mmcv.Config.fromfile(config_path)
    return config.model, config.data.test.pipeline, config.test_cfg


def torch2onnx(input, model):
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(
        model,
        input,
        'exported_model.onnx',
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': [0],
            'output': [0]
        })
    model = onnx.load('exported_onnx_model.onnx')
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))


def torch2caffe(input, model):
    try:
        import spring.nart.tools.pytorch as pytorch
    except ImportError as e:
        print(f'Cannot import nart tool: {e}')
        return
    with pytorch.convert_mode():
        pytorch.convert(
            model, [input],
            'exported_caffe_model',
            input_names=['input'],
            output_names=['output'])


if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
        checkpoint_path = sys.argv[2]
    except BaseException as e:
        print(f'{e}:\nPlease indicate the config file and checkpoint path.')
    model_cfg, test_pipeline, test_cfg = _get_recognizer_cfg(config_path)
    t = None
    s = None
    for trans in test_pipeline:
        if trans['type'] == 'SampleFrames':
            t = trans['clip_len']
        elif trans['type'] == 'Resize':
            if isinstance(trans['scale'], int):
                s = trans['scale']
            elif isinstance(trans['scale'], tuple):
                s = max(trans['scale'])

    dummy_input = torch.randn(1, 3, t, s, s).cuda()
    model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg).cuda()
    load_checkpoint(model, checkpoint_path)
    torch2onnx(dummy_input, model)
    torch2caffe(dummy_input, model)
