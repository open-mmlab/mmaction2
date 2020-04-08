import torch


def _get_cuda_home():
    if torch.__version__ == 'parrots':
        from parrots.utils.build_extension import CUDA_HOME
    else:
        from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


def get_build_config():
    if torch.__version__ == 'parrots':
        from parrots.config import get_build_info
        return get_build_info()
    else:
        return torch.__config__.show()


def _get_conv():
    if torch.__version__ == 'parrots':
        from parrots.nn.modules.conv import _ConvNd
    else:
        from torch.nn.modules.conv import _ConvNd
    return _ConvNd


def _get_batchnorm():
    if torch.__version__ == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.instancenorm import _InstanceNorm
        from torch.nn.modules.batchnorm import _BatchNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


CUDA_HOME = _get_cuda_home()
_ConvNd = _get_conv()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_batchnorm()


class SyncBatchNorm(SyncBatchNorm_):

    def _specify_ddp_gpu_num(self, gpu_size):
        if torch.__version__ == 'parrots':
            pass
        else:
            super()._specify_ddp_gpu_num(gpu_size)

    def _check_input_dim(self, input):
        if torch.__version__ == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)
