# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_layer_id_for_vit(var_name: str, max_layer_id: int) -> int:
    """Get the layer id to set the different learning rates for ViT.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.
    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id + 1


def get_layer_id_for_mvit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum layer id.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks'):
        layer_id = int(var_name.split('.')[2]) + 1
        return layer_id
    else:
        return max_layer_id + 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """
    Different learning rates are set for different layers of backbone.
    Note: Currently, this optimizer constructor is built for MViT.

    Inspiration from `the implementation in PySlowFast
    <https://github.com/facebookresearch/SlowFast>`_ and MMDetection
    <https://github.com/open-mmlab/mmdetection/tree/dev-3.x>`_
    """

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = MMLogger.get_current_instance()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers')
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build LearningRateDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd

        for m in module.modules():
            assert not isinstance(m, nn.modules.batchnorm._NormBase
                                  ), 'BN is not supported with layer decay'

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                if 'MViT' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_mvit(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                elif 'VisionTransformer' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_vit(name, num_layers)
                    logger.info(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError(f'Only support layer wise decay,'
                                          f'but got {decay_type}')

            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id + 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
