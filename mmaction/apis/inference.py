# Copyright (c) OpenMMLab. All rights reserved.
from operator import itemgetter
from typing import List, Optional, Tuple, Union

import mmengine
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner import load_checkpoint

from mmaction.registry import MODELS


def init_recognizer(config: Union[str, mmengine.Config],
                    checkpoint: Optional[str] = None,
                    device: Union[str, torch.device] = 'cuda:0') -> nn.Module:
    """Initialize a recognizer from config file.

    Args:
        config (Union[str, mmengine.Config]): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Defaults to None.
        device (Union[str, ``torch.device``]): The desired device of returned
            tensor. Defaults to ``cuda:0``.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, str):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    # pretrained model is unnecessary since we directly load checkpoint later
    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model: nn.Module,
                         video: str) -> List[Tuple[int, float]]:
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (str): The video file path.

    Returns:
        List[Tuple(int, float)]: Top-5 recognition result dict.
    """
    cfg = model.cfg

    # Build the data pipeline
    val_pipeline_cfg = cfg.val_dataloader.dataset.pipeline
    if 'Init' not in val_pipeline_cfg[0]['type']:
        val_pipeline_cfg = [dict(type='OpenCVInit')] + val_pipeline_cfg
    else:
        val_pipeline_cfg[0] = dict(type='OpenCVInit')
    for i in range(len(val_pipeline_cfg)):
        if 'Decode' in val_pipeline_cfg[i]['type']:
            val_pipeline_cfg[i] = dict(type='OpenCVDecode')
    val_pipeline = Compose(val_pipeline_cfg)

    # Prepare & process inputs
    data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    data = val_pipeline(data)
    data = pseudo_collate([data])

    # Forward the model
    with torch.no_grad():
        pred_scores = model.val_step(data)[0].pred_scores.item.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:5]

    return top5_label
