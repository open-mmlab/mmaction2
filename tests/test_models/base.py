import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import _BatchNorm


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def generate_backbone_demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs


def generate_recognizer_demo_inputs(
        input_shape=(1, 3, 3, 224, 224), model_type='2D'):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 250, 3, 224, 224).
        model_type (str): Model type for data generation, from {'2D', '3D'}.
            Default:'2D'
    """
    if len(input_shape) == 5:
        (N, L, C, H, W) = input_shape
    elif len(input_shape) == 6:
        (N, M, C, L, H, W) = input_shape

    imgs = np.random.random(input_shape)

    if model_type == '2D':
        gt_labels = torch.LongTensor([2] * N)
    elif model_type == '3D':
        gt_labels = torch.LongTensor([2] * M)
    elif model_type == 'audio':
        gt_labels = torch.LongTensor([2] * L)
    else:
        raise ValueError(f'Data type {model_type} is not available')

    inputs = {'imgs': torch.FloatTensor(imgs), 'gt_labels': gt_labels}
    return inputs


def generate_gradcam_inputs(input_shape=(1, 3, 3, 224, 224), model_type='2D'):
    """Create a superset of inputs needed to run gradcam.

    Args:
        input_shape (tuple[int]): input batch dimensions.
            Default: (1, 3, 3, 224, 224).
        model_type (str): Model type for data generation, from {'2D', '3D'}.
            Default:'2D'
    return:
        dict: model inputs, including two keys, ``imgs`` and ``label``.
    """
    imgs = np.random.random(input_shape)

    if model_type in ['2D', '3D']:
        gt_labels = torch.LongTensor([2] * input_shape[0])
    else:
        raise ValueError(f'Data type {model_type} is not available')

    inputs = {
        'imgs': torch.FloatTensor(imgs),
        'label': gt_labels,
    }
    return inputs


def get_cfg(config_type, fname):
    """Grab configs necessary to create a recognizer.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config_types = ('recognition', 'recognition_audio', 'localization')
    assert config_type in config_types

    repo_dpath = osp.dirname(osp.dirname(osp.dirname(__file__)))
    config_dpath = osp.join(repo_dpath, 'configs/' + config_type)
    config_fpath = osp.join(config_dpath, fname)
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    config = mmcv.Config.fromfile(config_fpath)
    return config


def get_recognizer_cfg(fname):
    return get_cfg('recognition', fname)


def get_audio_recognizer_cfg(fname):
    return get_cfg('recognition_audio', fname)


def get_localizer_cfg(fname):
    return get_cfg('localization', fname)
