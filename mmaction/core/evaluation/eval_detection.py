import json

import numpy as np

from .accuracy import interpolated_prec_rec, pairwise_temporal_iou


class ANetDetection(object):

    def __init__(self,
                 ground_truth_filename=None,
                 prediction_filename=None,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 verbose=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.ground_truth_filename = ground_truth_filename
        self.prediction_filename = prediction_filename
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded ground_truth from '
                  f'{self.ground_truth_filename}, prediction from '
                  f'{self.prediction_filename}.')
            num_gts = len(self.ground_truth)
            print(f'Number of ground truth instances: {num_gts}')
            num_preds = len(self.prediction)
            print(f'Number of predictions: {num_preds}')
            print(f'Fixed threshold for tiou score: {self.tiou_thresholds}')

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, returns the ground truth instances and the
        activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : list
            List containing the ground truth instances (dictionaries).
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as f:
            data = json.load(f)
        # Checking format
        activity_index, class_idx = {}, 0
        video_list, t_start_list, t_end_list, label_list = [], [], [], []
        for video_id, video_info in data.items():
            for anno in video_info['annotations']:
                if anno['label'] not in activity_index:
                    activity_index[anno['label']] = class_idx
                    class_idx += 1
                # old video_anno
                video_list.append(video_id[2:])
                t_start_list.append(float(anno['segment'][0]))
                t_end_list.append(float(anno['segment'][1]))
                label_list.append(activity_index[anno['label']])

        ground_truth = [{
            'video-id': vid,
            't-start': tst,
            't-end': ted,
            'label': lb
        } for vid, tst, ted, lb in zip(video_list, t_start_list, t_end_list,
                                       label_list)]

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, returns the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : list
            List containing the prediction instances (dictionaries).
        """
        with open(prediction_filename, 'r') as f:
            data = json.load(f)
        # Read predictions.
        video_list, t_start_list, t_end_list = [], [], []
        label_list, score_list = [], []
        for video_id, video_info in data['results'].items():
            for result in video_info:
                label = self.activity_index[result['label']]
                video_list.append(video_id)
                t_start_list.append(float(result['segment'][0]))
                t_end_list.append(float(result['segment'][1]))
                label_list.append(label)
                score_list.append(result['score'])
        prediction = [{
            'video-id': vid,
            't-start': tst,
            't-end': ted,
            'label': lb,
            'score': sc
        } for vid, tst, ted, lb, sc in zip(video_list, t_start_list,
                                           t_end_list, label_list, score_list)]
        return prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class."""
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = []
        prediction_by_label = []
        for i in range(len(self.activity_index)):
            ground_truth_by_label.append([])
            prediction_by_label.append([])
        for gt in self.ground_truth:
            ground_truth_by_label[gt['label']].append(gt)
        for pred in self.prediction:
            prediction_by_label[pred['label']].append(pred)

        results = [
            compute_average_precision_detection(
                ground_truth=ground_truth_by_label[i],
                prediction=prediction_by_label[i],
                tiou_thresholds=self.tiou_thresholds)
            for i in range(len(self.activity_index))
        ]

        for i in range(len(self.activity_index)):
            ap[:, i] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file.

        For the detection task we measure the interpolated mean average
        precision to measure the performance of a method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        return self.mAP, self.average_mAP


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : list
        List containing the ground truth instances (dictionaries).
        Required keys: ['video-id', 't-start', 't-end']
    prediction : list
        List containing the prediction instances (dictionaries).
        Required keys: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
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
    ground_truth_gbvn = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        if item['video-id'] not in ground_truth_gbvn:
            ground_truth_gbvn[item['video-id']] = []
        ground_truth_gbvn[item['video-id']].append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_gbvn:
            gts = ground_truth_gbvn[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        tiou_arr = pairwise_temporal_iou(
            np.array([pred['t-start'], pred['t-end']]),
            np.array([np.array([gt['t-start'], gt['t-end']]) for gt in gts]))
        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, gts[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, gts[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :],
                                         recall_cumsum[tidx, :])

    return ap
