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
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode', 'FrameSelector'
]


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('video', default='sample_video.mp4', help='video file')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('label', help='label file')
    parser.add_argument('--out-filename', default=None, help='out file name')
    parser.add_argument(
        '--input-step', type=int, default=1, help='internal between predict')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    args = parser.parse_args()
    return args


def show_results(fps=15):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    msg = ' '
    text_info = {}
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    frame_size = (frameWidth, frameHeight)
    ind = 0
    video_writer = cv2.VideoWriter(out_file, fourcc, fps, frame_size)
    while ind < frameCount:
        ind += 1
        ret, frame = cap.read()
        frame_queue.append(np.array(frame)[:, :, ::-1])
        ret, scores = inference()
        if ret is True:
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
                if score < threshold:
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


def inference():
    if len(frame_queue) != sample_length:
        return (False, None)
    cur_windows = list(np.array(frame_queue))
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue.popleft().shape[:2]
    for i in range(input_step):
        frame_queue.popleft()
    cur_data = data.copy()
    cur_data['imgs'] = cur_windows
    cur_data = test_pipeline(cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        cur_data = scatter(cur_data, [device])[0]
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]
    return True, scores


def main():
    global frame_queue, frame, results, threshold, sample_length, \
        data, test_pipeline, model, out_file, video_path, device, \
        input_step, label, result_queue
    args = parse_args()
    input_step = args.input_step
    threshold = args.threshold
    video_path = args.video
    out_file = args.out_file
    device = torch.device(args.device)
    model = init_recognizer(args.config, args.checkpoint, device=device)
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
    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        show_results()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
