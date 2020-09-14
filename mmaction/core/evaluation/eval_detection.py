import json

import numpy as np

from .accuracy import interpolated_prec_rec, segment_iou


class ANETdetection(object):

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
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(
                self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, returns the ground truth instances and the
        activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data.items():
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                # old video_anno
                video_lst.append(videoid[2:])
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = [{
            'video-id': vid,
            't-start': tst,
            't-end': ted,
            'label': lb
        } for vid, tst, ted, lb in zip(video_lst, t_start_lst, t_end_lst,
                                       label_lst)]

        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, returns the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = [{
            'video-id': vid,
            't-start': tst,
            't-end': ted,
            'label': lb,
            'score': sc
        } for vid, tst, ted, lb, sc in zip(video_lst, t_start_lst, t_end_lst,
                                           label_lst, score_lst)]
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name,
                                    cidx):
        """Get all predicitons of the given label.

        Return empty DataFrame if there is no predcitions with the given label.
        """
        return []
        if cidx in prediction_by_label.groups.keys():
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        else:
            print('Warning: No predictions of label \'%s\' were provdied.' %
                  label_name)
            return []

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
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    prediction = prediction.sort(key=lambda x: -x['score'])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = {}
    for item in ground_truth:
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

        tiou_arr = segment_iou(
            np.array([pred['t-start'], pred['t-end']]),
            np.array([np.array([gt['t-start'], gt['t-end']]) for gt in gts]))
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
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :],
                                         recall_cumsum[tidx, :])

    return ap
