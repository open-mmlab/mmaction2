# Copyright (c) OpenMMLab. All rights reserved.
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
        (N, L, _, _, _) = input_shape
    elif len(input_shape) == 6:
        (N, M, _, L, _, _) = input_shape

    imgs = np.random.random(input_shape)

    if model_type == '2D' or model_type == 'skeleton':
        gt_labels = torch.LongTensor([2] * N)
    elif model_type == '3D':
        gt_labels = torch.LongTensor([2] * M)
    elif model_type == 'audio':
        gt_labels = torch.LongTensor([2] * L)
    else:
        raise ValueError(f'Data type {model_type} is not available')

    inputs = {'imgs': torch.FloatTensor(imgs), 'gt_labels': gt_labels}
    return inputs


def generate_detector_demo_inputs(
        input_shape=(1, 3, 4, 224, 224), num_classes=81, train=True,
        device='cpu'):
    num_samples = input_shape[0]
    if not train:
        assert num_samples == 1

    def random_box(n):
        box = torch.rand(n, 4) * 0.5
        box[:, 2:] += 0.5
        box[:, 0::2] *= input_shape[3]
        box[:, 1::2] *= input_shape[4]
        if device == 'cuda':
            box = box.cuda()
        return box

    def random_label(n):
        label = torch.randn(n, num_classes)
        label = (label > 0.8).type(torch.float32)
        label[:, 0] = 0
        if device == 'cuda':
            label = label.cuda()
        return label

    img = torch.FloatTensor(np.random.random(input_shape))
    if device == 'cuda':
        img = img.cuda()

    proposals = [random_box(2) for i in range(num_samples)]
    gt_bboxes = [random_box(2) for i in range(num_samples)]
    gt_labels = [random_label(2) for i in range(num_samples)]
    img_metas = [dict(img_shape=input_shape[-2:]) for i in range(num_samples)]

    if train:
        return dict(
            img=img,
            proposals=proposals,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas)

    return dict(img=[img], proposals=[proposals], img_metas=[img_metas])


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
    config_types = ('recognition', 'recognition_audio', 'localization',
                    'detection', 'skeleton')
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


def get_detector_cfg(fname):
    return get_cfg('detection', fname)


def get_skeletongcn_cfg(fname):
    return get_cfg('skeleton', fname)
