import numpy as np

from ..registry import METRICS
from .base import BaseMetrics
from .overlaps import pairwise_temporal_iou
from .recall import binary_precision_recall_curve


def eval_ap(detections, gt_by_cls, iou_range):
    """Evaluate average precisions.

    Args:
        detections (dict): Results of detections.
        gt_by_cls (dict): Information of groundtruth.
        iou_range (list): Ranges of iou.

    Returns:
        list: Average precision values of classes at ious.
    """
    ap_values = np.zeros((len(detections), len(iou_range)))

    for iou_idx, min_overlap in enumerate(iou_range):
        for class_idx, _ in enumerate(detections):
            ground_truth = gt_by_cls[class_idx]
            new_ground_truth = []
            for video_id, segments in ground_truth.items():
                for segment in segments:
                    new_ground_truth.append({'video-id': video_id,
                                             't-start': segment[0],
                                             't-end': segment[1]})
            prediction = detections[class_idx]
            new_prediction = []
            for pred in prediction:
                new_prediction.append({'video-id': pred[0],
                                       't-start': pred[2],
                                       't-end': pred[3],
                                       'score': pred[4]})

            ap = average_precision_at_temporal_iou(new_ground_truth,
                                                   new_prediction,
                                                   [min_overlap])
            ap_values[class_idx, iou_idx] = ap

    return ap_values


def average_precision_at_temporal_iou(ground_truth,
                                      prediction,
                                      tiou_thresholds=np.linspace(
                                          0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', 't-end'
            and 'score'.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x['score'])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        tiou_arr = pairwise_temporal_iou(
            np.array([pred['t-start'], pred['t-end']]),
            np.array([np.array([gt['t-start'], gt['t-end']]) for gt in gts]))
        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return ap


def mmit_mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition. Used for reporting
    MMIT style mAP on Multi-Moments in Times. The difference is that this
    method calculates average-precision for each sample and averages them among
    samples.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The MMIT style mean average precision.
    """
    results = []
    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    return np.mean(results)


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


# TODO: check the name
def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


@METRICS.register_module()
class TemporalMeanAP(BaseMetrics):

    VALID_DATASETS = ('thumos14', )

    def __init__(self,
                 logger,
                 scale_range=np.arange(0.1, 1.0, .1),
                 eval_dataset='thumos14'):
        super().__init__(logger)

        assert eval_dataset in self.VALID_DATASETS

        self.scale_range = scale_range
        self.eval_dataset = eval_dataset

    def wrap_up(self, results):
        eval_results = {}
        for iou, map_iou in zip(self.scale_range, results):
            eval_results[f'mAP@{iou:.02f}'] = map_iou
        return eval_results

    def print_log_msg(self, log_msg):
        if self.logger is None:
            return
        super().print_log_msg(log_msg)
        self.logger.info('Evaluation finished')

    def __call__(self, results, gts, kwargs=None):
        if self.eval_dataset == 'thumos14':
            ap_values = eval_ap(results, gts, self.scale_range)
            map_ious = ap_values.mean(axis=0)
            eval_results = self.wrap_up(map_ious)

        self.print_log_msg(eval_results)
        return eval_results


@METRICS.register_module()
class MeanAP(BaseMetrics):

    VALID_MODE = (None, 'mit')

    def __init__(self, logger, mode=None):
        super().__init__(logger)

        assert mode in self.VALID_MODE
        self.mode = mode

    def wrap_up(self, results, kwargs):
        eval_results = {}
        if 'category' in kwargs:
            category = kwargs['category']
            eval_results[f'{category}_mAP'] = results
        else:
            eval_results['mean_average_precision'] = results
        return eval_results

    def __call__(self, results, gt_labels, kwargs=None):
        if self.mode == 'mit':
            mean_ap = mmit_mean_average_precision(results, gt_labels)
        elif self.mode is None:
            mean_ap = mean_average_precision(results, gt_labels)
        eval_results = self.wrap_up(mean_ap, kwargs)

        self.print_log_msg(eval_results)
        return eval_results
