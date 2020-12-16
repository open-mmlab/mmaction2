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
            ap = average_precision_at_temporal_iou(gt_by_cls[class_idx],
                                                   detections[class_idx],
                                                   [min_overlap])
            ap_values[class_idx, iou_idx] = ap

    return ap_values


def average_precision_at_temporal_iou(ground_truth,
                                      prediction,
                                      tiou_thresholds=(np.linspace(
                                          0.5, 0.95, 10))):
    """Compute average precision (in detection task) between ground truth and
    predicted data frames. If multiple predictions match the same predicted
    segment, only the one with highest score is matched as true positive. This
    code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
            Key: 'video_id'
            Value (np.ndarray): 1D array of 't-start' and 't-end'.
        prediction (np.ndarray): 2D array containing the information of
            proposal instances, including 'video_id', 'class_id', 't-start',
            't-end' and 'score'.
        tiou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        np.ndarray: 1D array of average precision score.
    """
    ap = np.zeros(len(tiou_thresholds), dtype=np.float32)
    if len(prediction) < 1:
        return ap

    num_gts = 0.
    lock_gt = dict()
    for key in ground_truth:
        lock_gt[key] = np.ones(
            (len(tiou_thresholds), len(ground_truth[key]))) * -1
        num_gts += len(ground_truth[key])

    # Sort predictions by decreasing score order.
    prediction = np.array(prediction)
    scores = prediction[:, 4].astype(float)
    sort_idx = np.argsort(scores)[::-1]
    prediction = prediction[sort_idx]

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)), dtype=np.int32)
    fp = np.zeros((len(tiou_thresholds), len(prediction)), dtype=np.int32)

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in enumerate(prediction):

        # Check if there is at least one ground truth in the video.
        if this_pred[0] in ground_truth:
            this_gt = np.array(ground_truth[this_pred[0]], dtype=float)
        else:
            fp[:, idx] = 1
            continue

        t_iou = pairwise_temporal_iou(this_pred[2:4].astype(float), this_gt)
        # We would like to retrieve the predictions with highest t_iou score.
        t_iou_sorted_idx = t_iou.argsort()[::-1]
        for t_idx, t_iou_threshold in enumerate(tiou_thresholds):
            for jdx in t_iou_sorted_idx:
                if t_iou[jdx] < t_iou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[this_pred[0]][t_idx, jdx] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[this_pred[0]][t_idx, jdx] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float32)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float32)
    recall_cumsum = tp_cumsum / num_gts

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
