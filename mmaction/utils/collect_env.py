from mmcv.utils import collect_env as collect_basic_env

import mmaction


def collect_env():
    env_info = collect_basic_env()
    env_info['MMAction2'] = mmaction.__version__
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
