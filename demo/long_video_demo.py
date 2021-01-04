import argparse
import random
from collections import deque
from operator import itemgetter

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 predict different labels in a long video demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video_path', help='video file/url')
    parser.add_argument('label', help='label file')
    parser.add_argument('out_file', help='output filename')
    parser.add_argument(
        '--input-step',
        type=int,
        default=1,
        help='input step for sampling frames')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--stride',
        type=float,
        default=0,
        help=('the prediction stride equals to stride * sample_length '
              '(sample_length indicates the size of temporal window from '
              'which you sample frames, which equals to '
              'clip_len x frame_interval), if set as 0, the '
              'prediction stride is 1'))
    args = parser.parse_args()
    return args


def show_results(model, data, label, args):
    frame_queue = deque(maxlen=args.sample_length)
    result_queue = deque(maxlen=1)

    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    msg = 'Preparing action recognition ...'
    text_info = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frame_width, frame_height)

    ind = 0
    video_writer = cv2.VideoWriter(args.out_file, fourcc, fps, frame_size)
    prog_bar = mmcv.ProgressBar(num_frames)
    backup_frames = []

    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = cap.read()
        if frame is None:
            # drop it when encounting None
            continue
        backup_frames.append(np.array(frame)[:, :, ::-1])
        if ind == args.sample_length:
            # provide a quick show at the beginning
            frame_queue.extend(backup_frames)
            backup_frames = []
        elif ((len(backup_frames) == args.input_step
               and ind > args.sample_length) or ind == num_frames):
            # pick a frame from the backup
            # when the backup is full or reach the last frame
            chosen_frame = random.choice(backup_frames)
            backup_frames = []
            frame_queue.append(chosen_frame)

        ret, scores = inference(model, data, args, frame_queue)

        if ret:
            num_selected_labels = min(len(label), 5)
            scores_tuples = tuple(zip(label, scores))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < args.threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
        elif len(text_info):
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)
        video_writer.write(frame)
    cap.release()
    cv2.destroyAllWindows()


def inference(model, data, args, frame_queue):
    if len(frame_queue) != args.sample_length:
        # Do no inference when there is no enough frames
        return False, None

    cur_windows = list(np.array(frame_queue))
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]

    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = args.test_pipeline(cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [args.device])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]

    if args.stride > 0:
        pred_stride = int(args.sample_length * args.stride)
        for i in range(pred_stride):
            frame_queue.popleft()

    # for case ``args.stride=0``
    # deque will automatically popleft one element

    return True, scores


def main():
    args = parse_args()

    args.device = torch.device(args.device)
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0
    args.sample_length = sample_length
    args.test_pipeline = test_pipeline

    show_results(model, data, label, args)


if __name__ == '__main__':
    main()
