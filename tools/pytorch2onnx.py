import argparse

import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_localizer, build_recognizer

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=v1.0.4')


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None):
    model.cpu().eval()

    one_img = torch.randn(input_shape)

    # onnx.export does not support kwargs
    if hasattr(model, 'dummy_forward'):
        model.forward = model.dummy_forward
    elif hasattr(model, '_forward') and args.is_localizer:
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    register_extra_symbolics(opset_version)
    torch.onnx.export(
        model, ([one_img]),
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=show,
        opset_version=opset_version)

    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model([one_img])

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(None,
                               {net_feed_input[0]: one_img.detach().numpy()})
        # only compare a part of result
        assert np.allclose(
            pytorch_result[0][:, 4], onnx_result[:, 4]
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMAction models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        help='whether it is a localizer')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 8, 224, 224],
        help='input video size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMAction2 only support opset 11 now'

    normalize_cfg = {
        'mean': np.array(args.mean, dtype=np.float32),
        'std': np.array(args.std, dtype=np.float32)
    }

    cfg = mmcv.Config.fromfile(args.config)
    # import modules from string list.

    cfg.model.pretrained = None

    # build the model
    if args.is_localizer:
        model = build_localizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    else:
        model = build_recognizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model = _convert_batchnorm(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2onnx(
        model,
        args.input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg)
