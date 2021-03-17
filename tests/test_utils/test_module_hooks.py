import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch

from mmaction.models import build_recognizer
from mmaction.utils import register_module_hooks
from mmaction.utils.module_hooks import GPUNormalize


def test_register_module_hooks():
    _module_hooks = [
        dict(
            type='GPUNormalize',
            hooked_module='backbone',
            hook_pos='forward_pre',
            input_format='NCHW',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375])
    ]

    repo_dpath = osp.dirname(osp.dirname(osp.dirname(__file__)))
    config_fpath = osp.join(repo_dpath, 'configs/_base_/models/tsm_r50.py')
    config = mmcv.Config.fromfile(config_fpath)
    config.model['backbone']['pretrained'] = None

    # case 1
    module_hooks = copy.deepcopy(_module_hooks)
    module_hooks[0]['hook_pos'] = 'forward_pre'
    recognizer = build_recognizer(config.model)
    handles = register_module_hooks(recognizer, module_hooks)
    assert recognizer.backbone._forward_pre_hooks[
        handles[0].id].__name__ == 'normalize_hook'

    # case 2
    module_hooks = copy.deepcopy(_module_hooks)
    module_hooks[0]['hook_pos'] = 'forward'
    recognizer = build_recognizer(config.model)
    handles = register_module_hooks(recognizer, module_hooks)
    assert recognizer.backbone._forward_hooks[
        handles[0].id].__name__ == 'normalize_hook'

    # case 3
    module_hooks = copy.deepcopy(_module_hooks)
    module_hooks[0]['hooked_module'] = 'cls_head'
    module_hooks[0]['hook_pos'] = 'backward'
    recognizer = build_recognizer(config.model)
    handles = register_module_hooks(recognizer, module_hooks)
    assert recognizer.cls_head._backward_hooks[
        handles[0].id].__name__ == 'normalize_hook'

    # case 4
    module_hooks = copy.deepcopy(_module_hooks)
    module_hooks[0]['hook_pos'] = '_other_pos'
    recognizer = build_recognizer(config.model)
    with pytest.raises(ValueError):
        handles = register_module_hooks(recognizer, module_hooks)

    # case 5
    module_hooks = copy.deepcopy(_module_hooks)
    module_hooks[0]['hooked_module'] = '_other_module'
    recognizer = build_recognizer(config.model)
    with pytest.raises(ValueError):
        handles = register_module_hooks(recognizer, module_hooks)


def test_gpu_normalize():

    def check_normalize(origin_imgs, result_imgs, norm_cfg):
        """Check if the origin_imgs are normalized correctly into result_imgs
        in a given norm_cfg."""
        from numpy.testing import assert_array_almost_equal
        target_imgs = result_imgs.copy()
        target_imgs *= norm_cfg['std']
        target_imgs += norm_cfg['mean']
        assert_array_almost_equal(origin_imgs, target_imgs, decimal=4)

    _gpu_normalize_cfg = dict(
        input_format='NCTHW',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375])

    # case 1
    gpu_normalize_cfg = copy.deepcopy(_gpu_normalize_cfg)
    gpu_normalize_cfg['input_format'] = 'NCHW'
    gpu_normalize = GPUNormalize(**gpu_normalize_cfg)
    assert gpu_normalize._mean.shape == (1, 3, 1, 1)
    imgs = np.random.randint(256, size=(2, 240, 320, 3), dtype=np.uint8)
    _input = (torch.tensor(imgs).permute(0, 3, 1, 2), )
    normalize_hook = gpu_normalize.hook_func()
    _input = normalize_hook(torch.nn.Module, _input)
    result_imgs = np.array(_input[0].permute(0, 2, 3, 1))
    check_normalize(imgs, result_imgs, gpu_normalize_cfg)

    # case 2
    gpu_normalize_cfg = copy.deepcopy(_gpu_normalize_cfg)
    gpu_normalize_cfg['input_format'] = 'NCTHW'
    gpu_normalize = GPUNormalize(**gpu_normalize_cfg)
    assert gpu_normalize._mean.shape == (1, 3, 1, 1, 1)

    # case 3
    gpu_normalize_cfg = copy.deepcopy(_gpu_normalize_cfg)
    gpu_normalize_cfg['input_format'] = 'NCHW_Flow'
    gpu_normalize = GPUNormalize(**gpu_normalize_cfg)
    assert gpu_normalize._mean.shape == (1, 3, 1, 1)

    # case 4
    gpu_normalize_cfg = copy.deepcopy(_gpu_normalize_cfg)
    gpu_normalize_cfg['input_format'] = 'NPTCHW'
    gpu_normalize = GPUNormalize(**gpu_normalize_cfg)
    assert gpu_normalize._mean.shape == (1, 1, 1, 3, 1, 1)

    # case 5
    gpu_normalize_cfg = copy.deepcopy(_gpu_normalize_cfg)
    gpu_normalize_cfg['input_format'] = '_format'
    with pytest.raises(ValueError):
        gpu_normalize = GPUNormalize(**gpu_normalize_cfg)
