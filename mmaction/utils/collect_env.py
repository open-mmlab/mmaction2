# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_basic_env

import mmaction


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_basic_env()
    env_info['MMAction2'] = (
        mmaction.__version__ + '+' + get_git_hash(digits=7))
    env_info['MMCV'] = (mmcv.__version__)

    try:
        import mmdet
        env_info['MMDetection'] = (mmdet.__version__)
    except ImportError:
        pass

    try:
        import mmpose
        env_info['MMPose'] = (mmpose.__version__)
    except ImportError:
        pass

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
