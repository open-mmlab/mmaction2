import os.path as osp
import pickle
from collections import defaultdict

import numpy as np

from .bbox_overlaps import iou2d, spatio_temporal_iou3d, spatio_temporal_nms3d


def pr_to_ap(precision_recall):

    pr_diff = precision_recall[1:, 1] - precision_recall[:-1, 1]
    pr_sum = precision_recall[1:, 0] + precision_recall[:-1, 0]

    return np.sum(pr_diff * pr_sum * 0.5)


def frame_ap(det_results, labels, videos, gt_tubes, threshold=0.5):
    results = []
    for label_index, label in enumerate(labels):
        det_result = det_results[det_results[:, 2] == label_index, :]

        gt = defaultdict(list)
        for video_id, video in enumerate(videos):
            tubes = gt_tubes[video]

            if label_index not in tubes:
                continue

            for tube in tubes[label_index]:
                for i in range(len(tube)):
                    key = (video_id, int(tube[i, 0]))
                    gt[key].append(tube[i, 1:5].tolist())

        for key in gt:
            gt[key] = np.array(gt[key].copy())

        precision_recall = np.empty((det_result.shape[0] + 1, 2),
                                    dtype=np.float32)
        precision_recall[0, 0] = 1.0
        precision_recall[0, 1] = 0.0
        fn, fp, tp = sum([item.shape[0] for item in gt.values()]), 0, 0

        for i, j in enumerate(np.argsort(-det_result[:, 3])):
            key = (int(det_result[j, 0]), int(det_result[j, 1]))
            box = det_result[j, 4:8]
            is_positive = False

            if key in gt:
                ious = iou2d(gt[key], box)
                max_idx = np.argmax(ious)

                if ious[max_idx] >= threshold:
                    is_positive = True
                    gt[key] = np.delete(gt[key], max_idx, 0)

                    if gt[key].size == 0:
                        del gt[key]

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision_recall[i + 1, 0] = tp / (tp + fp)
            precision_recall[i + 1, 1] = tp / (tp + fn)

        results.append(pr_to_ap(precision_recall))

    frame_ap_result = np.mean(results * 100)
    return frame_ap_result


def frame_ap_error(det_results, labels, videos, gt_tubes, threshold=0.5):
    ap_results = []
    other_ap_results = [[], [], [], []]
    missing_detections = []
    for label_index, label in enumerate(labels):
        det_result = det_results[det_results[:, 2] == label_index, :]

        gt = defaultdict(list)
        other_gt = defaultdict(list)
        label_dict = defaultdict(list)

        for video_id, video in enumerate(videos):
            tubes = gt_tubes[video]
            label_dict[video_id] = list(tubes)

            for tube_label_index in tubes:
                for tube in tubes[tube_label_index]:
                    for i in range(tube.shape[0]):
                        key = (video_id, int(tube[i, 0]))
                        if tube_label_index == label_index:
                            gt[key].append(tube[i, 1:5].tolist())
                        else:
                            other_gt[key].append(tube[i, 1:5].tolist())

        for key in gt:
            gt[key] = np.array(gt[key].copy())
        for key in other_gt:
            other_gt[key] = np.array(other_gt[key].copy())

        original_key = list(gt)

        precision_recall = np.empty((det_result.shape[0] + 1, 6),
                                    dtype=np.float32)
        precision_recall[0, 0] = 1.0
        precision_recall[0, 1:] = 0.0

        fn = sum([item.shape[0] for item in gt.values()])
        (fp, tp, localization_error, classification_error, other_error,
         time_error) = (0, 0, 0, 0, 0, 0)

        for i, j in enumerate(np.argsort(-det_result[:, 3])):
            key = (int(det_result[j, 0]), int(det_result[j, 1]))
            box = det_result[j, 4:8]
            is_positive = False

            if key in original_key:
                if key in gt:
                    ious = iou2d(gt[key], box)
                    max_idx = np.argmax(ious)

                    if ious[max_idx] >= threshold:
                        is_positive = True
                        gt[key] = np.delete(gt[key], max_idx, 0)

                        if gt[key].size == 0:
                            del gt[key]
                    else:
                        localization_error += 1
                else:
                    localization_error += 1

            elif key in other_gt:
                ious = iou2d(other_gt[key], box)
                if np.max(ious) >= threshold:
                    classification_error += 1
                else:
                    other_error += 1

            elif label_index in label_dict[key[0]]:
                time_error += 1
            else:
                other_error += 1

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision_recall[i + 1, 0] = tp / (tp + fp)
            precision_recall[i + 1, 1] = tp / (tp + fn)
            precision_recall[i + 1, 2] = localization_error / (tp + fp)
            precision_recall[i + 1, 3] = classification_error / (tp + fp)
            precision_recall[i + 1, 4] = time_error / (tp + fp)
            precision_recall[i + 1, 5] = other_error / (tp + fp)

        ap_results.append(pr_to_ap(precision_recall[..., :2]))
        for j in range(2, 6):
            other_ap_results[j - 2].append(
                pr_to_ap(precision_recall[..., [j, 1]]))
        missing_detections.append(precision_recall[-1, 1])

    ap_results = np.array(ap_results) * 100
    other_ap_results = np.array(other_ap_results) * 100

    (localization_error, classification_error, time_error,
     other_error) = other_ap_results[:4]
    missing_detections = 100 - 100 * np.array(missing_detections)

    result = dict(
        ap_results=ap_results,
        localization_error=localization_error,
        classification_error=classification_error,
        time_error=time_error,
        other_error=other_error,
        missing_detections=missing_detections)

    result_str = ''
    for i, label in enumerate(labels):
        result_str += f'{label:20s}' + ' '.join(
            [f'{v[i]:8.2f}' for v in result.values()]) + '\n'
    result_str += '\n' + f"{'mean':20s}" + ' '.join(
        [f'{np.mean(v):8.2f}' for v in result.values()]) + '\n'

    msg = 'Error Analysis\n'
    msg += f"\n{'label':20s} {'   AP   ':8s} {'  Loc.  ':8s} {'  Cls.  ':8s} "
    msg += f"{'  Time  ':8s} {' Other ':8s} {' missed ':8s}\n"
    msg += f'\n{result_str}'

    print(msg)

    return result


def video_ap(labels, videos, gt_tubes, tube_dir, threshold=0.5, overlap=0.3):

    det_results = defaultdict(list)

    for video in videos:
        tube_name = osp.join(tube_dir, video + '_tubes.pkl')
        if not osp.isfile(tube_name):
            raise FileNotFoundError(f'Extracted tubes {tube_name} is missing')

        with open(tube_name, 'rb') as f:
            tubes = pickle.load(f)

        for label_index in range(len(labels)):
            tube = tubes[label_index]
            index = spatio_temporal_nms3d(tube, overlap)
            det_results[label_index].extend([(video, tube[i][1], tube[i][0])
                                             for i in index])

    results = []
    for label_index in range(len(labels)):
        det_result = np.array(det_results[label_index])

        gt = defaultdict(list)
        for video in videos:
            tubes = gt_tubes[video]

            if label_index not in tubes:
                continue

            gt[video] = tubes[label_index].copy()
            if len(gt[video]) == 0:
                del gt[video]

        precision_recall = np.empty((len(det_result) + 1, 2), dtype=np.float32)
        precision_recall[0, 0] = 1.0
        precision_recall[0, 1] = 0.0
        fn, fp, tp = sum([len(item) for item in gt.values()]), 0, 0

        dets = -np.array(det_result[:, 1])
        for i, j in enumerate(np.argsort(dets)):
            key, score, tube = det_result[j]
            is_positive = False

            if key in gt:
                ious = [spatio_temporal_iou3d(g, tube) for g in gt[key]]
                max_index = np.argmax(ious)
                if ious[max_index] >= threshold:
                    is_positive = True
                    del gt[key][max_index]

                    if len(gt[key]) == 0:
                        del gt[key]

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            precision_recall[i + 1, 0] = tp / (tp + fp)
            precision_recall[i + 1, 1] = tp / (tp + fn)

        results.append(pr_to_ap(precision_recall))

    video_ap_result = np.mean(results * 100)
    return video_ap_result
