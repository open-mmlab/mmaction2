# Copyright (c) OpenMMLab. All rights reserved.
import time

import cv2
import numpy as np
import torch
from mmengine.dataset import Compose
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

from mmaction.apis import inference_recognizer, init_recognizer

label_names = [
    'Doing other things',
    'Drumming Fingers',
    'No gesture',  # 0
    'Pulling Hand In',
    'Pulling Two Fingers In',
    'Pushing Hand Away',
    'Pushing Two Fingers Away',  # 3
    'Rolling Hand Backward',
    'Rolling Hand Forward',
    'Shaking Hand',  # 7
    'Sliding Two Fingers Down',
    'Sliding Two Fingers Left',  # 10
    'Sliding Two Fingers Right',
    'Sliding Two Fingers Up',  # 12
    'Stop Sign',
    'Swiping Down',
    'Swiping Left',
    'Swiping Right',
    'Swiping Up',  # 14
    'Dislike',
    'Like',
    'Turning Hand Clockwise',
    'Turning Hand Counterclockwise',  # 19
    'Zooming In With Full Hand',
    'Zooming In With Two Fingers',  # 23
    'Zooming Out With Full Hand',
    'Zooming Out With Two Fingers',  # 25
    'Call',
    'Fist',
    'Four',
    'Mute',
    'OK',
    'One',
    'Palm',  # 27
    'Peace',
    'Rock',
    'Three-Middle',
    'Three-Left',
    'Two Up',
    'No Gesture'  # 34
]


class HandVisualizer:

    _RED = (48, 48, 255)
    _GREEN = (48, 255, 48)
    _BLUE = (192, 101, 21)
    _YELLOW = (0, 204, 255)
    _GRAY = (128, 128, 128)
    _PURPLE = (128, 64, 128)
    _PEACH = (180, 229, 255)
    _WHITE = (224, 224, 224)

    keypoint_info = {
        0: dict(name='wrist', id=0, color=_RED),
        1: dict(name='thumb1', id=1, color=_RED),
        2: dict(name='thumb2', id=2, color=_PEACH),
        3: dict(name='thumb3', id=3, color=_PEACH),
        4: dict(name='thumb4', id=4, color=_PEACH),
        5: dict(name='forefinger1', id=5, color=_RED),
        6: dict(name='forefinger2', id=6, color=_PURPLE),
        7: dict(name='forefinger3', id=7, color=_PURPLE),
        8: dict(name='forefinger4', id=8, color=_PURPLE),
        9: dict(name='middle_finger1', id=9, color=_RED),
        10: dict(name='middle_finger2', id=10, color=_YELLOW),
        11: dict(name='middle_finger3', id=11, color=_YELLOW),
        12: dict(name='middle_finger4', id=12, color=_YELLOW),
        13: dict(name='ring_finger1', id=13, color=_RED),
        14: dict(name='ring_finger2', id=14, color=_GREEN),
        15: dict(name='ring_finger3', id=15, color=_GREEN),
        16: dict(name='ring_finger4', id=16, color=_GREEN),
        17: dict(name='pinky_finger1', id=17, color=_RED),
        18: dict(name='pinky_finger2', id=18, color=_BLUE),
        19: dict(name='pinky_finger3', id=19, color=_BLUE),
        20: dict(name='pinky_finger4', id=20, color=_BLUE)
    }

    skeleton_info = {
        0: dict(link=('wrist', 'thumb1'), id=0, color=_GRAY),
        1: dict(link=('thumb1', 'thumb2'), id=1, color=_PEACH),
        2: dict(link=('thumb2', 'thumb3'), id=2, color=_PEACH),
        3: dict(link=('thumb3', 'thumb4'), id=3, color=_PEACH),
        4: dict(link=('wrist', 'forefinger1'), id=4, color=_GRAY),
        5: dict(link=('forefinger1', 'forefinger2'), id=5, color=_PURPLE),
        6: dict(link=('forefinger2', 'forefinger3'), id=6, color=_PURPLE),
        7: dict(link=('forefinger3', 'forefinger4'), id=7, color=_PURPLE),
        8: dict(link=('wrist', 'middle_finger1'), id=8, color=_GRAY),
        9:
        dict(link=('middle_finger1', 'middle_finger2'), id=9, color=_YELLOW),
        10:
        dict(link=('middle_finger2', 'middle_finger3'), id=10, color=_YELLOW),
        11:
        dict(link=('middle_finger3', 'middle_finger4'), id=11, color=_YELLOW),
        12: dict(link=('wrist', 'ring_finger1'), id=12, color=_GRAY),
        13: dict(link=('ring_finger1', 'ring_finger2'), id=13, color=_GREEN),
        14: dict(link=('ring_finger2', 'ring_finger3'), id=14, color=_GREEN),
        15: dict(link=('ring_finger3', 'ring_finger4'), id=15, color=_GREEN),
        16: dict(link=('wrist', 'pinky_finger1'), id=16, color=_GRAY),
        17: dict(link=('pinky_finger1', 'pinky_finger2'), id=17, color=_BLUE),
        18: dict(link=('pinky_finger2', 'pinky_finger3'), id=18, color=_BLUE),
        19: dict(link=('pinky_finger3', 'pinky_finger4'), id=19, color=_BLUE)
    }

    def __init__(self, radius=5, line_width=3):
        self.radius = radius
        self.radius_border = max(radius + 1, int(radius * 1.2))
        self.line_width = line_width

        self.keypoint_id2name = dict()
        self.keypoint_name2id = dict()
        self.keypoint_colors = []
        self.skeleton_links = []
        self.skeleton_link_colors = []

        for kpt_id, kpt in self.keypoint_info.items():
            kpt_name = kpt['name']
            self.keypoint_id2name[kpt_id] = kpt_name
            self.keypoint_name2id[kpt_name] = kpt_id
            self.keypoint_colors.append(kpt['color'])

        for sk in self.skeleton_info.values():
            self.skeleton_links.append(
                tuple(self.keypoint_name2id[n] for n in sk['link']))
            self.skeleton_link_colors.append(sk['color'])

    def draw_keypoints(self, img, keypoints):
        for kpts in keypoints:
            for sk_id, sk in enumerate(self.skeleton_links):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                color = self.skeleton_link_colors[sk_id]
                cv2.line(img, pos1, pos2, color, thickness=self.line_width)

            for kid, kpt in enumerate(kpts):
                color = self.keypoint_colors[kid]
                x_coord, y_coord = int(kpt[0]), int(kpt[1])
                cv2.circle(img, (int(x_coord), int(y_coord)),
                           self.radius_border, self._WHITE, -1)
                cv2.circle(img, (int(x_coord), int(y_coord)), self.radius,
                           color, -1)


def kp2box(kpt, margin=0.2):
    min_x, max_x = min(kpt[:, 0]), max(kpt[:, 0])
    min_y, max_y = min(kpt[:, 1]), max(kpt[:, 1])
    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    w, h = max_x - min_x, max_y - min_y
    w2, h2 = w * (1 + margin) / 2, h * (1 + margin) / 2
    min_x = max(0, c_x - w2)
    min_y = max(0, c_y - h2)
    max_x = min(1, c_x + w2)
    max_y = min(1, c_y + h2)
    return min_x, min_y, max_x - min_x, max_y - min_y


def flip_box(box):
    return 1 - box[0] - box[2], box[1], box[2], box[3]


def h2r(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def create_fake_anno(history, keypoint, bbox, clip_len=10):
    from mmdet.models.task_modules import BboxOverlaps2D
    bbox = torch.tensor(bbox)[None]
    iou_calc = BboxOverlaps2D()
    results = [keypoint]

    # frame contains tuples of (keypoint, bbox)
    for frame in history[::-1]:
        anchors = torch.tensor([x[1] for x in frame])
        if anchors.shape[0] == 0:
            break
        ious = iou_calc(bbox, anchors)[0]
        idx = torch.argmax(ious)
        if ious[idx] >= 0.5:
            results.append(frame[idx][0])
            bbox = anchors[idx:idx + 1]
        else:
            break
        if len(results) >= clip_len:
            break

    keypoint = np.array(results[::-1], dtype=np.float32)[None]
    total_frames = keypoint.shape[1]
    return dict(
        keypoint=keypoint,
        total_frames=total_frames,
        frame_dir='NA',
        label=0,
        start_index=0,
        modality='Pose')


def main():
    pose_model = init_model(
        'configs/rtmpose-m.py', 'checkpoints/rtmpose-m.pth', device='cpu')
    recognizer = init_recognizer(
        'configs/stgcnpp.py', 'checkpoints/hagrid.pth', device='cpu')
    cfg = recognizer.cfg
    test_pipeline = Compose(cfg.test_pipeline)

    keypoints_buffer = []
    results_buffer = []
    frame_idx = 0
    predict_per_nframe = 2
    plate = 'FE6506-FC5915-F94929-F73F36-F32653-F11867-EE0A77'.split('-')
    plate = [h2r(x)[::-1] for x in plate]

    visualizer = HandVisualizer()

    # For webcam input:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while cap.isOpened():
        success, image = cap.read()
        frame_idx += 1

        if not success:
            print('Ignoring empty camera frame.')
            continue

        # Inference a single image.
        batch_results = inference_topdown(pose_model, image)
        results = merge_data_samples(batch_results)
        raw_keypoints = results.pred_instances.keypoints

        # Draw the hand annotations on the image.
        visualizer.draw_keypoints(image, raw_keypoints)

        boxes = []
        keypoints = []

        h, w, _ = image.shape
        raw_keypoints[..., 0] = raw_keypoints[..., 0] / w
        raw_keypoints[..., 1] = raw_keypoints[..., 1] / h
        for kpts in raw_keypoints:
            box = kp2box(kpts)
            boxes.append(box)
            keypoints.append((kpts, box))

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        # Draw the boxes on the image.
        for box in boxes:
            box = flip_box(box)
            x, y, w, h = [int(v * s) for v, s in zip(box, (w, h, w, h))]
            cv2.rectangle(
                image, (x, y), (x + w, y + h),
                color=(0, 0, 255),
                thickness=w // 120)

        if frame_idx % predict_per_nframe == 0:
            if len(keypoints) == 0:
                results_buffer.append('No hand detected')
            else:
                for keypoint, bbox in keypoints:
                    with torch.no_grad():
                        sample = create_fake_anno(keypoints_buffer, keypoint,
                                                  bbox)
                        prediction = inference_recognizer(
                            recognizer, sample, test_pipeline=test_pipeline)
                        action = prediction.pred_labels.item
                        scores = prediction.pred_scores.item
                        action_name = label_names[action]
                        results_buffer.append(
                            f'{action_name}: {scores[action].item():.3f}')

        FONTFACE = cv2.FONT_HERSHEY_DUPLEX
        FONTSCALE = 0.6
        THICKNESS = 1
        LINETYPE = 1
        for i, (action_label,
                color) in enumerate(zip(results_buffer[::-1][:7], plate)):
            cv2.putText(image, action_label, (10, 24 + i * 24), FONTFACE,
                        FONTSCALE, color, THICKNESS, LINETYPE)

        cv2.putText(
            image, f'FPS: {round(frame_idx / (time.time() - start_time), 4)}',
            (10, 8 * 24), FONTFACE, FONTSCALE, (255, 0, 220), THICKNESS,
            LINETYPE)

        keypoints_buffer.append(keypoints)

        cv2.imshow('MMAction2 Gesture Demo [Press ESC to Exit]', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == '__main__':
    main()
