# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
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
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    while True:
        msg = 'Waiting for action ...'
        _, frame = camera.read()
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = collate([cur_data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            cur_data = scatter(cur_data, [device])[0]

        with torch.no_grad():
            scores = model(return_loss=False, **cur_data)[0]

        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

        if inference_fps > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()


def main():
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        result_queue, drawing_fps, inference_fps

    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps

    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(cfg, args.checkpoint, device=device)
    camera = cv2.VideoCapture(args.camera_id)
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.data.test.pipeline
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

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
