"""Webcam Spatio-Temporal Action Detection Demo.

This Script borrows some codes from
https://github.com/facebookresearch/SlowFast # noqa
"""

import argparse
import atexit
import copy
import queue
import threading
import time

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_detector
from mmaction.utils import import_module_error_func

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass


class TaskInfo:
    """Wapper for a clip."""

    def __init__(self):
        # raw frames
        self.frames = None
        # preprocessed(resize/norm) frames
        self.processed_frames = None
        self.id = -1
        self.bboxes = None
        self.action_preds = None
        self.num_buffer_frames = 0
        self.clip_vis_radius = -1
        self.img_shape = None

    def add_frames(self, idx, frames, processed_frames):
        """Add the clip and corresponding id.

        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
        """
        self.frames = frames
        self.processed_frames = processed_frames
        self.id = idx
        self.img_shape = processed_frames[0].shape[:2]

    def add_bboxes(self, bboxes):
        """Add correspondding bounding boxes."""
        self.bboxes = bboxes

    def add_action_preds(self, preds):
        """Add the corresponding action predictions."""
        self.action_preds = preds

    def get_model_inputs(self, device):
        input_array = np.stack(self.processed_frames).transpose(
            (3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(device)
        return dict(
            return_loss=False,
            img=[input_tensor],
            proposals=[[self.bboxes]],
            img_metas=[[dict(img_shape=self.img_shape)]])


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--config',
        default=('configs/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py'),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.5,
        help='the threshold of human action score')
    parser.add_argument('--video', help='video file/url')
    parser.add_argument(
        '--label-map', default='demo/label_map_ava.txt', help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument('--clip-vis-radius', default=5, type=int, help='')
    parser.add_argument(
        '--output-fps',
        default=30,
        type=int,
        help='the fps of demo video output')
    args = parser.parse_args()
    return args


class BaseHumanDetector:

    def _do_detect(self, image):
        """Get human bboxes with shape [n, 4].

        The format of bboxes is (xmin, ymin, xmax, ymax) in pixels.
        """
        raise NotImplementedError

    def predict(self, task):
        keyframe = task.frames[len(task.frames) // 2]
        bboxes = self._do_detect(keyframe)
        task.add_bboxes(bboxes)
        return task


class MmdetHumanDetector(BaseHumanDetector):

    def __init__(self, config, ckpt, device, score_thr, person_classid=0):
        self.model = init_detector(config, ckpt, device)
        self.person_classid = person_classid
        self.score_thr = score_thr
        self.device = device

    def _do_detect(self, image):
        result = inference_detector(self.model, image)[self.person_classid]
        result = result[result[:, 4] >= self.score_thr][:, :4]
        result = torch.from_numpy(result).to(self.device)
        return result


class StdetPredictor:

    def __init__(self, config, checkpoint, device, score_thr, label_map_path):
        # load model
        config.model.backbone.pretrained = None
        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location=device)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

        self.score_thr = score_thr
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}

    def predict(self, task):
        if len(task.bboxes) == 0:
            return task

        # model inference
        with torch.no_grad():
            # result for one sample, a list with num_classes elements
            # each element
            result = self.model(**task.get_model_inputs(self.device))[0]

        # post process
        preds = []
        for i in range(task.bboxes.shape[0]):
            preds.append([])
        for class_id in range(len(result)):
            if class_id + 1 not in self.label_map:
                continue
            for bboex_id in range(task.bboxes.shape[0]):
                if result[class_id][bboex_id, 4] > self.score_thr:
                    preds[bboex_id].append((self.label_map[class_id + 1],
                                            result[class_id][bboex_id, 4]))
        task.add_action_preds(preds)

        return task


class ClipHelper:
    """Multithrading utils to read/show frames and create TaskInfo object."""

    def __init__(self, config, video_path, predict_stepsize, output_fps,
                 clip_vis_radius):
        # init source
        try:
            self.cap = cv2.VideoCapture(int(video_path))
        except ValueError:
            self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened()
        was_read, frame = self.cap.read()
        assert was_read

        # image meta & preprocess params
        h, w, _ = frame.shape
        self.origin_size = (h, w)
        self.new_size = mmcv.rescale_size((w, h), (256, np.Inf))
        self.ratio = (n / o for n, o in zip(self.new_size, self.origin_size))
        img_norm_cfg = config['img_norm_cfg']
        if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
            to_bgr = img_norm_cfg.pop('to_bgr')
            img_norm_cfg['to_rgb'] = to_bgr
        img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
        img_norm_cfg['std'] = np.array(img_norm_cfg['std'])
        self.img_norm_cfg = img_norm_cfg

        # sampling strategy
        val_pipeline = config['val_pipeline']
        sampler = [x for x in val_pipeline
                   if x['type'] == 'SampleAVAFrames'][0]
        clip_len, frame_interval = sampler['clip_len'], sampler[
            'frame_interval']
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'

        # init params
        self.clip_vis_radius = clip_vis_radius
        self.output_fps = output_fps
        self.predict_stepsize = predict_stepsize
        self.window_size = clip_len * frame_interval
        self.buffer_size = self.window_size - self.predict_stepsize
        assert self.buffer_size < self.window_size // 2

        self.display_id = -1
        self.read_id = -1
        self.buffer = []
        self.processed_buffer = []
        self.read_queue = queue.Queue()
        self.display_queue = {}
        self.display_lock = threading.Lock()
        self.read_id_lock = threading.Lock()
        self.read_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.not_end = True
        self.stopped = False

        atexit.register(self.clean)

    def read_fn(self):
        """Read frames from VideoCapture and create tasks."""
        was_read = True
        while was_read and not self.stopped:
            # create task
            task = TaskInfo()
            task.clip_vis_radius = self.clip_vis_radius

            # read frames to create a clip
            frames = []
            processed_frames = []
            if len(self.buffer) != 0:
                frames = self.buffer
            if len(self.processed_buffer) != 0:
                processed_frames = self.processed_buffer
            with self.read_lock:
                while was_read and len(frames) < self.window_size:
                    was_read, frame = self.cap.read()
                    if was_read:
                        frame = mmcv.imresize(frame, self.new_size)
                        frames.append(frame)
                        processed_frame = frame.astype(np.float32)
                        _ = mmcv.imnormalize_(processed_frame,
                                              **self.img_norm_cfg)
                        processed_frames.append(processed_frame)
            if was_read:
                self.buffer = frames[-self.buffer_size:]
                self.processed_buffer = processed_frames[-self.buffer_size:]

            task.add_frames(self.read_id + 1, frames, processed_frames)
            task.num_buffer_frames = (0 if self.read_id == -1 else
                                      self.buffer_size)
            with self.read_id_lock:
                self.read_id += 1
                self.not_end = was_read

            self.read_queue.put((was_read, copy.deepcopy(task)))

    def display_fn(self):
        while not self.stopped:
            with self.read_id_lock:
                read_id = self.read_id
                not_end = self.not_end

            with self.display_lock:
                # If video ended and we have display all frames.
                if not not_end and self.display_id == read_id:
                    break
                # If the next frames are not available, wait.
                if (len(self.display_queue) == 0 or
                        self.display_queue.get(self.display_id + 1) is None):
                    time.sleep(0.02)
                    continue
                else:
                    self.display_id += 1
                    was_read, task = self.display_queue[self.display_id]
                    del self.display_queue[self.display_id]

            with self.output_lock:
                for frame in task.frames[task.num_buffer_frames:]:
                    cv2.imshow('Demo', frame)
                    cv2.waitKey(int(1000 / self.output_fps))

    def __iter__(self):
        return self

    def __next__(self):
        if self.read_queue.qsize() == 0:
            time.sleep(0.02)
            return True, None
        else:
            return self.read_queue.get()

    def start(self):
        """Start threads to read and display frames."""
        self.read_thread = threading.Thread(
            target=self.read_fn, args=(), name='VidRead-Thread', daemon=True)
        self.read_thread.start()
        self.display_thread = threading.Thread(
            target=self.display_fn,
            args=(),
            name='VidDisplay-Thread',
            daemon=True)
        self.display_thread.start()

        return self

    def clean(self):
        self.stopped = True
        self.read_lock.acquire()
        self.cap.release()
        self.read_lock.release()
        self.output_lock.acquire()
        cv2.destroyAllWindows()
        self.output_lock.release()

    def join(self):
        self.read_thread.join()
        self.display_thread.join()

    def display(self, task):
        """Add the visualized task to the display queue for display.

        Args:
            task (TaskInfo object): task object that contain the necessary
            information for prediction visualization.
        """
        with self.display_lock:
            self.display_queue[task.id] = (True, task)


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


class Visualizer:
    """plate (str): The plate used for visualization.

    Default: plate_blue.
    max_labels_per_bbox (int): Max number of labels to visualize for a person
        box. Default: 5.
    """

    def __init__(self,
                 plate='03045e-023e8a-0077b6-0096c7-00b4d8-48cae4',
                 max_labels_per_bbox=5):

        def hex2color(h):
            """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
            return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

        plate = plate.split('-')
        self.plate = [hex2color(h) for h in plate]
        self.max_labels_per_bbox = max_labels_per_bbox

    def draw_predictions(self, task):
        bboxes = task.bboxes.cpu().numpy()
        frames = task.frames
        preds = task.action_preds

        # already displayed by the former task
        buffer = frames[:task.num_buffer_frames]

        # display with preds
        keyframe_idx = len(frames) // 2 - task.num_buffer_frames
        draw_range = [
            keyframe_idx - task.clip_vis_radius,
            keyframe_idx + task.clip_vis_radius,
        ]
        frames = self.draw_clip_range(frames[task.num_buffer_frames:], preds,
                                      bboxes, keyframe_idx, draw_range)
        task.frames = buffer + frames

        return task

    def draw_clip_range(self, frames, preds, bboxes, keyframe_idx, draw_range):
        if bboxes is None or len(bboxes) == 0:
            return frames

        if draw_range is None:
            draw_range = [0, len(frames) - 1]
        else:
            draw_range[0] = max(0, draw_range[0])
        left_frames = frames[:draw_range[0]]
        right_frames = frames[draw_range[1] + 1:]
        draw_frames = frames[draw_range[0]:draw_range[1] + 1]

        # draw bboxes and texts
        labels = []
        for bbox_preds in preds:
            labels.append([x[0] for x in bbox_preds])
        h, w, _ = frames[0].shape
        scale_ratio = np.array([w, h, w, h])
        for frame in draw_frames:
            self.draw_one_image(frame, bboxes, labels, scale_ratio)

        return list(left_frames) + draw_frames + list(right_frames)

    def draw_one_image(self, frame, bboxes, labels, scale_ratio):
        for bbox, label in zip(bboxes, labels):
            box = bbox.astype(np.int64)
            st, ed = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(frame, st, ed, (0, 0, 255), 2)
            for k, text in enumerate(label):
                if k >= self.max_labels_per_bbox:
                    break
                location = (0 + st[0], 18 + k * 18 + st[1])
                textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                           THICKNESS)[0]
                textwidth = textsize[0]
                diag0 = (location[0] + textwidth, location[1] - 14)
                diag1 = (location[0], location[1] + 2)
                cv2.rectangle(frame, diag0, diag1, self.plate[k + 1], -1)
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)


def main(args):
    # init visualizer
    vis = Visualizer()

    # init human detector
    human_detector = MmdetHumanDetector(args.det_config, args.det_checkpoint,
                                        args.device, args.det_score_thr)

    # init action detector
    config = mmcv.Config.fromfile(args.config)

    # Fix a issue that different actions may have different bboxes
    try:
        config['model']['test_cfg']['rcnn']['action_thr'] = .0
    except KeyError:
        pass

    stdet_predictor = StdetPredictor(
        config=config,
        checkpoint=args.checkpoint,
        device=args.device,
        score_thr=args.action_score_thr,
        label_map_path=args.label_map)

    # init clip helper
    clip_helper = ClipHelper(
        config=config,
        video_path=args.video,
        predict_stepsize=args.predict_stepsize,
        output_fps=args.output_fps,
        clip_vis_radius=args.clip_vis_radius)
    clip_helper.start()

    for able_to_read, task in clip_helper:
        if task is None:
            time.sleep(0.02)
            continue
        human_detector.predict(task)
        stdet_predictor.predict(task)
        vis.draw_predictions(task)
        clip_helper.display(task)

        if not able_to_read:
            break

    clip_helper.join()
    clip_helper.clean()


if __name__ == '__main__':
    main(parse_args())
