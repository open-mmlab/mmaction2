import argparse

import mmcv
from mmcv import Config, DictAction

from mmaction.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    assert args.eval is not None

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)

    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule',
            'key_indicator'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metrics=args.eval, **kwargs))
    print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
