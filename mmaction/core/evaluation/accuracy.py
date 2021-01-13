import numpy as np

from ..registry import METRICS
from .base import BaseMetrics


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def get_weighted_score(score_list, coeff_list):
    """Get weighted score with given scores and coefficients.

    Given n predictions by different classifier: [score_1, score_2, ...,
    score_n] (score_list) and their coefficients: [coeff_1, coeff_2, ...,
    coeff_n] (coeff_list), return weighted score: weighted_score =
    score_1 * coeff_1 + score_2 * coeff_2 + ... + score_n * coeff_n

    Args:
        score_list (list[list[np.ndarray]]): List of list of scores, with shape
            n(number of predictions) X num_samples X num_classes
        coeff_list (list[float]): List of coefficients, with shape n.

    Returns:
        list[np.ndarray]: List of weighted scores.
    """
    assert len(score_list) == len(coeff_list)
    num_samples = len(score_list[0])
    for i in range(1, len(score_list)):
        assert len(score_list[i]) == num_samples

    scores = np.array(score_list)  # (num_coeff, num_samples, num_classes)
    coeff = np.array(coeff_list)  # (num_coeff, )
    weighted_scores = list(np.dot(scores.T, coeff).T)
    return weighted_scores


@METRICS.register_module()
class TopKAcc(BaseMetrics):

    def __init__(self, logger, topk=(1, 5)):
        super().__init__(logger)

        if not isinstance(topk, (int, tuple)):
            raise TypeError('topk must be int or tuple of int, '
                            f'but got {type(topk)}')

        if isinstance(topk, int):
            topk = (topk, )
        self.topk = topk

    def wrap_up(self, results):
        eval_results = {}
        for k, v in zip(self.topk, results):
            eval_results[f'top{k}_acc'] = v
        return eval_results

    def __call__(self, results, gt_labels, kwargs=None):
        acc = top_k_accuracy(results, gt_labels, self.topk)
        eval_results = self.wrap_up(acc)

        self.print_log_msg(eval_results)
        return eval_results


@METRICS.register_module()
class MeanClsAcc(BaseMetrics):

    def wrap_up(self, results):
        eval_results = {'mean_class_accuracy': results}
        return eval_results

    def __call__(self, results, gt_labels, kwargs=None):
        mean_cls_acc = mean_class_accuracy(results, gt_labels)
        eval_results = self.wrap_up(mean_cls_acc)

        self.print_log_msg(eval_results)
        return eval_results
