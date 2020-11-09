import argparse
import os
import os.path as osp

import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.utils.gradcam_utils import GradCAM


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 GradCAM demo')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--target-layer-name',
        type=str,
        default='backbone/layer4/1/relu',
        help='GradCAM target layer name')
    parser.add_argument('--out-filename', default=None, help='output filename')
    parser.add_argument('--fps', default=5, type=int)

    args = parser.parse_args()
    return args


def build_inputs(model, video_path, use_frames=False):
    """build inputs, codes from `inference_recognizer`"""
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    elif osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

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

    return data


def main():
    args = parse_args()

    # assign the desired device.
    device = torch.device(args.device)

    # build the recognizer from a config file and checkpoint file
    model = init_recognizer(
        args.config,
        args.checkpoint,
        device=device,
        use_frames=args.use_frames)

    inputs = build_inputs(model, args.video, use_frames=args.use_frames)
    gradcam = GradCAM(model, args.target_layer_name)
    results = gradcam(inputs)

    if args.out_filename is not None:
        from moviepy.editor import ImageSequenceClip

        # frames_batches shape [B, T, H, W, 3], in RGB order
        frames_batches = (results[0] * 255.).numpy().astype(np.uint8)
        frames = frames_batches.reshape(-1, *frames_batches.shape[-3:])
        frame_list = [frame for frame in frames]
        video_clips = ImageSequenceClip(frame_list, fps=args.fps)
        out_type = osp.splitext(args.out_filename)[1][1:]
        if out_type == 'gif':
            video_clips.write_gif(args.out_filename)
        else:
            video_clips.write_videofile(args.out_filename, remove_temp=True)


if __name__ == '__main__':
    main()
