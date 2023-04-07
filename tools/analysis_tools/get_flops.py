# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine import Config
from mmengine.registry import init_default_scope

from mmaction.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    elif len(args.shape) == 4:
        # n, c, h, w = args.shape for 2D recognizer
        input_shape = tuple(args.shape)
    elif len(args.shape) == 5:
        # n, c, t, h, w = args.shape for 3D recognizer or
        # n, m, t, v, c = args.shape for GCN-based recognizer
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmaction'))
    model = MODELS.build(cfg.model)
    model.eval()

    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    analysis_results = get_model_complexity_info(model, input_shape)
    flops = analysis_results['flops_str']
    params = analysis_results['params_str']
    table = analysis_results['out_table']
    print(table)
    split_line = '=' * 30
    print(f'\n{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
