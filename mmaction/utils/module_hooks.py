import torch
from mmcv.utils import Registry, build_from_cfg

MODULE_HOOKS = Registry('module_hooks')


def register_module_hooks(Module, module_hooks_list):
    for module_hook_cfg in module_hooks_list:
        Module.register_forward_pre_hook(
            build_from_cfg(module_hook_cfg, MODULE_HOOKS).hook_func())


@MODULE_HOOKS.register_module()
class GPUNormalize:
    """Normalize images with the given mean and std value on GPUs.

    Call the member function ``hook_func`` will return the forward pre-hook
    function for module registration.

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
    """

    def __init__(self, input_format, mean, std):
        if input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(f'The input format {input_format} is invalid.')
        self.input_format = input_format
        _mean = torch.tensor(mean)
        _std = torch.tensor(std)
        if input_format == 'NCTHW':
            self._mean = _mean[None, :, None, None, None]
            self._std = _std[None, :, None, None, None]
        elif input_format == 'NCHW':
            self._mean = _mean[None, :, None, None]
            self._std = _std[None, :, None, None]
        elif input_format == 'NCHW_Flow':
            self._mean = _mean[None, :, None, None]
            self._std = _std[None, :, None, None]
        elif input_format == 'NPTCHW':
            self._mean = _mean[None, None, None, :, None, None]
            self._std = _std[None, None, None, :, None, None]

    def hook_func(self):

        def normalize_forward_pre_hook(Module, input):
            x = input[0]
            if not hasattr(self, 'mean'):
                self.mean = self._mean.to(x.device)
                self.std = self._std.to(x.device)

            assert self.mean.device == x.device
            with torch.no_grad():
                x.sub_(self.mean).div_(self.std)

        return normalize_forward_pre_hook
