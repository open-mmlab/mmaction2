import os
import os.path as osp
from operator import itemgetter

import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from ..datasets.pipelines import Compose
from ..models import build_recognizer


def init_recognizer(config,
                    checkpoint=None,
                    device='cuda:0',
                    use_frames=False):
    """Initialize a recognizer from config file.

    Args:
        config (str | :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str | None, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Default: None.
        device (str | :obj:`torch.device`): The desired device of returned
            tensor. Default: 'cuda:0'.
        use_frames (bool): Whether to use rawframes as input. Default:False.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if ((use_frames and config.dataset_type != 'RawframeDataset')
            or (not use_frames and config.dataset_type != 'VideoDataset')):
        input_type = 'rawframes' if use_frames else 'video'
        raise RuntimeError('input data type should be consist with the '
                           f'dataset type in config, but got input type '
                           f"'{input_type}' and dataset type "
                           f"'{config.dataset_type}'")
    # pretrained model is unnecessary since we directly load checkpoint later
    config.model.backbone.pretrained = None
    model = build_recognizer(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=device)
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model, video_path, label_path, use_frames=False):
    """Inference a video with the detector.

    Args:
        model (nn.Module): The loaded recognizer.
        video_path (str): The video file path/url or the rawframes directory
            path. If ``use_frames`` is set to True, it should be rawframes
            directory path. Otherwise, it should be video file path.
        label_path (str): The label file path.
        use_frames (bool): Whether to use rawframes as input. Default:False.

    Returns:
        dict[tuple(str, float)]: Top-5 recognition result dict.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # construct label map
    with open(label_path, 'r') as f:
        label = [line.strip() for line in f]
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if use_frames:
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=len(os.listdir(video_path)),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
    else:
        start_index = cfg.data.test.get('start_index', 0)
        data = dict(
            filename=video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
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
