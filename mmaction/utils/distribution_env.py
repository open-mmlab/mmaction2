# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

dp_factory = {
        'cuda' : MMDataParallel
    }

ddp_factory = {
        'cuda' : MMDistributedDataParallel
    }

def build_dp(model, device = 'cuda'):
    assert device in ['cuda', 'mlu'], "Only available for cuda or mlu devices."
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return model, dp_factory[device]

def build_ddp(model, device = 'cuda'):
    assert device in ['cuda', 'mlu'], "Only available for cuda or mlu devices."
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import  MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return model, ddp_factory[device]

def select_device():
    is_device_available = {'cuda': torch.cuda.is_available(),
                       'mlu': hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()}
    st_device = [k for k, v in is_device_available.items() if v ]
    assert len(st_device) == 1, "Only one device type is available, please check!"
    device = st_device[0]
    return device

default_device = select_device()

def current_device():
    idx = ''
    if default_device=='mlu':
        idx = torch.mlu.current_device()
    elif default_device=='cuda':
        idx = torch.cuda.current_device()
    return idx

