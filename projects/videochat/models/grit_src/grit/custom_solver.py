# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Jialian Wu from https://github.com/facebookresearch/Detic/blob
# /main/detic/custom_solver.py
import itertools
from typing import Any, Dict, List, Set

import torch
from detectron2.config import CfgNode
from detectron2.solver.build import maybe_add_gradient_clipping


def build_custom_optimizer(cfg: CfgNode,
                           model: torch.nn.Module) -> torch.optim.Optimizer:
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    optimizer_type = cfg.SOLVER.OPTIMIZER

    for key, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            continue
        # Avoid duplicating parameters
        if value in memo:
            continue
        memo.add(value)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if cfg.SOLVER.VIT_LAYER_DECAY:
            lr = lr * get_vit_lr_decay_rate(
                key, cfg.SOLVER.VIT_LAYER_DECAY_RATE, cfg.MODEL.VIT_LAYERS)

        param = {'params': [value], 'lr': lr}
        if optimizer_type != 'ADAMW':
            param['weight_decay'] = weight_decay
        params += [param]

    def maybe_add_full_model_gradient_clipping(
            optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == 'full_model'
            and clip_norm_val > 0.0)

        class FullModelGradientClippingOptimizer(optim):

            def step(self, closure=None):
                all_params = itertools.chain(
                    *[x['params'] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    if optimizer_type == 'SGD':
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV)
    elif optimizer_type == 'ADAMW':
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise NotImplementedError(f'no optimizer type {optimizer_type}')
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == 'full_model':
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith('backbone'):
        if '.pos_embed' in name or '.patch_embed' in name:
            layer_id = 0
        elif '.blocks.' in name and '.residual.' not in name:
            layer_id = int(name[name.find('.blocks.'):].split('.')[2]) + 1

    return lr_decay_rate**(num_layers + 1 - layer_id)
