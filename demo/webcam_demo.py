import argparse
from collections import deque
from operator import itemgetter

import cv2
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
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
        default=0,
        help='recognition score threshold')
    parser.add_argument(
        '--sample-length',
        type=int,
        default=0,
        help='number of sampled frames')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    args = parser.parse_args()
    return args


def predict_webcam_video():
    data = dict(img_shape=None, modality='RGB')

    windows = deque()
    score_cache = deque()
    scores_sum = np.zeros(len(label))

    while True:
        ret, frame = camera.read()
        # BGR to RGB
        windows.append(np.array(frame[:, :, ::-1]))
        if data['img_shape'] is None:
            data['img_shape'] = frame.shape[:2]

        cur_windows = list(np.array(windows))
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

        if len(windows) == sample_length:
            windows.popleft()
        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

            scores_sum -= score_cache.popleft()
            cv2.imshow('camera', frame)
            ch = cv2.waitKey(1)

            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    camera.release()
    cv2.destroyAllWindows()


def main():
    global label, device, model, test_pipeline, \
        camera, sample_length, average_size, threshold

    args = parse_args()
    device = torch.device(args.device)
    model = init_recognizer(args.config, args.checkpoint, device=device)
    camera = cv2.VideoCapture(args.camera_id)

    sample_length = args.sample_length
    average_size = args.average_size
    threshold = args.threshold

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            # Remove step to sample frames
            if sample_length == 0:
                sample_length = step['clip_len'] * step['num_clips']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    print('Press "Esc", "q" or "Q" to exit')
    predict_webcam_video()


if __name__ == '__main__':
    main()
