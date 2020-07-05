from operator import itemgetter

import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from ..datasets.pipelines import Compose
from ..models import build_recognizer


def init_recognizer(config, checkpoint=None, device='cuda:0'):
    """Initialize a recognizer from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Default: None.
        device (str or :obj:`torch.device`): the desired device of returned
            tensor. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # pretrained model is unnecessary since we directly load checkpoint later
    config.model.backbone.pretrained = None
    model = build_recognizer(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=device)
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model, video_path, label_path):
    """Inference a video with the detector.

    Args:
        model (nn.Module): The loaded recognizer.
        video_path (str): The video file path.
        label_path (str): The label file path.

    Returns:
        dict[tuple(str, float)]: Top-5 recognition result dict.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # construct label map
    with open(label_path, 'r') as f:
        label = [line.strip() for line in f]
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(filename=video_path, label=label, modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)[0]
    score_tuples = tuple(zip(label, scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

    top5_label = score_sorted[:5]
    return top5_label
