import copy

import torch


def gpu_normalize(input_format,
                  mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375]):
    if input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
        raise ValueError(f'The input format {input_format} is invalid.')
    _mean = torch.tensor(copy.deepcopy(mean))
    _std = torch.tensor(copy.deepcopy(std))

    if input_format == 'NCTHW':
        _mean = _mean[None, :, None, None, None]
        _std = _std[None, :, None, None, None]
    elif input_format == 'NCHW':
        _mean = _mean[None, :, None, None]
        _std = _std[None, :, None, None]
    elif input_format == 'NCHW_Flow':
        _mean = _mean[None, :, None, None]
        _std = _std[None, :, None, None]
    elif input_format == 'NPTCHW':
        _mean = _mean[None, None, None, :, None, None]
        _std = _std[None, None, None, :, None, None]

    def normalize_forward_pre_hook(Module, input):
        x = input[0]
        if not hasattr(normalize_forward_pre_hook, 'mean'):
            normalize_forward_pre_hook.mean = _mean.to(x.device)
            normalize_forward_pre_hook.std = _std.to(x.device)

        with torch.no_grad():
            x.sub_(normalize_forward_pre_hook.mean).div_(
                normalize_forward_pre_hook.std)

    return normalize_forward_pre_hook
