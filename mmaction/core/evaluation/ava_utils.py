import csv
import heapq
import logging
import time
from collections import defaultdict

import numpy as np

from .ava_evaluation import object_detection_evaluation as det_eval
from .ava_evaluation import standard_fields
from .recall import eval_recalls


def det2csv(dataset, results):
    csv_results = []
    for idx in range(len(dataset)):
        video_id = dataset.video_infos[idx]['video_id']
        timestamp = dataset.video_infos[idx]['timestamp']
        result = results[idx]
        for label, _ in enumerate(result):
            for bbox in result[label]:
                bbox_ = tuple(bbox.tolist())
                csv_results.append((
                    video_id,
                    timestamp,
                ) + bbox_[:4] + (label + 1, ) + bbox_[4:])
    return csv_results


# results is organized by class
def results2csv(dataset, results, out_file):
    if isinstance(results[0], list):
        csv_results = det2csv(dataset, results)

    # save space for float
    def tostr(item):
        if isinstance(item, float):
            return f'{item:.3f}'
        return str(item)

    with open(out_file, 'w') as f:
        for csv_result in csv_results:
            f.write(','.join(map(lambda x: tostr(x), csv_result)))
            f.write('\n')


def print_time(message, start):
    print('==> %g seconds to %s' % (time.time() - start, message))


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return f'{video_id},{int(timestamp):04d}'


def read_csv(csv_file, class_whitelist=None, capacity=0):
    """Loads boxes and class labels from a CSV file in the AVA format.

    CSV file format described at https://research.google.com/ava/download.html.

    Args:
        csv_file: A file object.
        class_whitelist: If provided, boxes corresponding to (integer) class
        labels not in this set are skipped.
        capacity: Maximum number of labeled boxes allowed for each example.
        Default is 0 where there is no limit.

    Returns:
        boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
        labels: A dictionary mapping each unique image key (string) to a list
        of integer class lables, matching the corresponding box in `boxes`.
        scores: A dictionary mapping each unique image key (string) to a list
        of score values lables, matching the corresponding label in `labels`.
        If scores are not provided in the csv, then they will default to 1.0.
    """
    start = time.time()
    entries = defaultdict(list)
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row) in [7, 8], 'Wrong number of columns: ' + row
        image_key = make_image_key(row[0], row[1])
        x1, y1, x2, y2 = [float(n) for n in row[2:6]]
        action_id = int(row[6])
        if class_whitelist and action_id not in class_whitelist:
            continue

        score = 1.0
        if len(row) == 8:
            score = float(row[7])
        if capacity < 1 or len(entries[image_key]) < capacity:
            heapq.heappush(entries[image_key],
                           (score, action_id, y1, x1, y2, x2))
        elif score > entries[image_key][0][0]:
            heapq.heapreplace(entries[image_key],
                              (score, action_id, y1, x1, y2, x2))
    for image_key in entries:
        # Evaluation API assumes boxes with descending scores
        entry = sorted(entries[image_key], key=lambda tup: -tup[0])
        for item in entry:
            score, action_id, y1, x1, y2, x2 = item
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    print_time('read file ' + csv_file.name, start)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.

    Args:
        exclusions_file: A file object containing a csv of video-id,timestamp.

    Returns:
        A set of strings containing excluded image keys, e.g.
        "aaaaaaaaaaa,0904",
        or an empty set if exclusions file is None.
    """
    excluded = set()
    if exclusions_file:
        reader = csv.reader(exclusions_file)
    for row in reader:
        assert len(row) == 2, 'Expected only 2 columns, got: ' + row
        excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Reads a labelmap without the dependency on protocol buffers.

    Args:
        labelmap_file: A file object containing a label map protocol buffer.

    Returns:
        labelmap: The label map in the form used by the
        object_detection_evaluation
        module - a list of {"id": integer, "name": classname } dicts.
        class_ids: A set containing all of the valid class id integers.
    """
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids


# Seems there is at most 100 detections for each image
def ava_eval(result_file,
             result_type,
             label_file,
             ann_file,
             exclude_file,
             max_dets=(100, ),
             verbose=True):

    assert result_type in ['mAP']

    start = time.time()
    categories, class_whitelist = read_labelmap(open(label_file))

    # loading gt, do not need gt score
    gt_boxes, gt_labels, _ = read_csv(open(ann_file), class_whitelist, 0)
    if verbose:
        print_time('Reading detection results', start)

    if exclude_file is not None:
        excluded_keys = read_exclusions(open(exclude_file))
    else:
        excluded_keys = list()

    start = time.time()
    boxes, labels, scores = read_csv(open(result_file), class_whitelist, 0)
    if verbose:
        print_time('Reading detection results', start)

    if result_type == 'proposal':
        gts = [
            np.array(gt_boxes[image_key], dtype=float)
            for image_key in gt_boxes
        ]
        proposals = []
        for image_key in gt_boxes:
            if image_key in boxes:
                proposals.append(
                    np.concatenate(
                        (np.array(boxes[image_key], dtype=float),
                         np.array(scores[image_key], dtype=float)[:, None]),
                        axis=1))
            else:
                # if no corresponding proposal, add a fake one
                proposals.append(np.array([0, 0, 1, 1, 1]))

        # Proposals used here are with scores
        recalls = eval_recalls(gts, proposals, np.array(max_dets),
                               np.arange(0.5, 0.96, 0.05))
        ar = recalls.mean(axis=1)
        ret = {}
        for i, num in enumerate(max_dets):
            print(f'Recall@0.5@{num}\t={recalls[i, 0]:.4f}')
            print(f'AR@{num}\t={ar[i]:.4f}')
            ret[f'Recall@0.5@{num}'] = recalls[i, 0]
            ret[f'AR@{num}'] = ar[i]
        return ret

    if result_type == 'mAP':
        pascal_evaluator = det_eval.PascalDetectionEvaluator(categories)

        start = time.time()
        for image_key in gt_boxes:
            if verbose and image_key in excluded_keys:
                logging.info(
                    'Found excluded timestamp in detections: %s.'
                    'It will be ignored.', image_key)
                continue
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(gt_boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                    np.array(gt_labels[image_key], dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(gt_boxes[image_key]), dtype=bool)
                })
        if verbose:
            print_time('Convert groundtruth', start)

        start = time.time()
        for image_key in boxes:
            if verbose and image_key in excluded_keys:
                logging.info(
                    'Found excluded timestamp in detections: %s.'
                    'It will be ignored.', image_key)
                continue
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float)
                })
        if verbose:
            print_time('convert detections', start)

        start = time.time()
        metrics = pascal_evaluator.evaluate()
        if verbose:
            print_time('run_evaluator', start)
        for display_name in metrics:
            print(f'{display_name}=\t{metrics[display_name]}')
        return {
            display_name: metrics[display_name]
            for display_name in metrics if 'ByCategory' not in display_name
        }
