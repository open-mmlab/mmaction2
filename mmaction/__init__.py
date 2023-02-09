# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__

mmcv_minimum_version = '2.0.0rc0'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.5.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (digit_version(mmcv_minimum_version) <= mmcv_version
        < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (digit_version(mmengine_minimum_version) <= mmengine_version
        < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

__all__ = ['__version__']
