import mmcv
from mmcv import digit_version

from .version import __version__

mmcv_minimum_version = '1.3.1'
mmcv_maximum_version = '1.4.0'
mmcv_version = digit_version(mmcv.__version__)

assert (digit_version(mmcv_minimum_version) <= mmcv_version
        <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

__all__ = ['__version__']
