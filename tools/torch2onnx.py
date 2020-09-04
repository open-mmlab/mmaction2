import os.path as osp
import sys

import mmcv
import onnx
import torch

from mmaction.models import build_model


def _get_recognizer_cfg(config_path):
    """Grab configs necessary to create a recognizer."""
    if not osp.exists(config_path):
        raise Exception('Cannot find config path')
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
    model = onnx.load('exported_model.onnx')
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))


if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
    except BaseException:
        print('Please indicate the config file path.')
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

    dummy_input = torch.randn(8, 3, t, s, s).cuda()
    model = build_model(model_cfg, train_cfg=None, test_cfg=test_cfg).cuda()
    torch2onnx(dummy_input, model)
