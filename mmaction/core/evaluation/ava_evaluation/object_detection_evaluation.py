# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""object_detection_evaluation module.

ObjectDetectionEvaluation is a class which manages ground truth information of
a object detection dataset, and computes frequently used detection metrics such
as Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.

Note: This module operates on numpy boxes and box lists.
"""

import collections
import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np

from . import metrics, per_image_evaluation, standard_fields


class DetectionEvaluator:
    """Interface for object detection evalution classes.

    Example usage of the Evaluator:
    ------------------------------
    evaluator = DetectionEvaluator(categories)

    # Detections and groundtruth for image 1.
    evaluator.add_single_groundtruth_image_info(...)
    evaluator.add_single_detected_image_info(...)

    # Detections and groundtruth for image 2.
    evaluator.add_single_groundtruth_image_info(...)
    evaluator.add_single_detected_image_info(...)

    metrics_dict = evaluator.evaluate()
    """

    __metaclass__ = ABCMeta

    def __init__(self, categories):
        """Constructor.

        Args:
            categories: A list of dicts, each of which has the following keys -
                'id': (required) an integer id uniquely identifying this
                    category.
                'name': (required) string representing category name e.g.,
                    'cat', 'dog'.
        """
        self._categories = categories

    @abstractmethod
    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary of groundtruth numpy arrays required
                for evaluations.
        """

    @abstractmethod
    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary of detection numpy arrays required
                for evaluation.
        """

    @abstractmethod
    def evaluate(self):
        """Evaluates detections and returns a dictionary of metrics."""

    @abstractmethod
    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""


class ObjectDetectionEvaluator(DetectionEvaluator):
    """A class to evaluate detections."""

    def __init__(self,
                 categories,
                 matching_iou_threshold=0.5,
                 evaluate_corlocs=False,
                 metric_prefix=None,
                 use_weighted_mean_ap=False,
                 evaluate_masks=False):
        """Constructor.

        Args:
            categories: A list of dicts, each of which has the following keys -
                'id': (required) an integer id uniquely identifying this
                    category.
                'name': (required) string representing category name e.g.,
                    'cat', 'dog'.
            matching_iou_threshold: IOU threshold to use for matching
                groundtruth boxes to detection boxes.
            evaluate_corlocs: (optional) boolean which determines if corloc
                scores are to be returned or not.
            metric_prefix: (optional) string prefix for metric name; if None,
                no prefix is used.
            use_weighted_mean_ap: (optional) boolean which determines if the
                mean average precision is computed directly from the scores and
                tp_fp_labels of all classes.
            evaluate_masks: If False, evaluation will be performed based on
                boxes. If True, mask evaluation will be performed instead.

        Raises:
            ValueError: If the category ids are not 1-indexed.
        """
        super(ObjectDetectionEvaluator, self).__init__(categories)
        self._num_classes = max([cat['id'] for cat in categories])
        if min(cat['id'] for cat in categories) < 1:
            raise ValueError('Classes should be 1-indexed.')
        self._matching_iou_threshold = matching_iou_threshold
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._label_id_offset = 1
        self._evaluate_masks = evaluate_masks
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids = set([])
        self._evaluate_corlocs = evaluate_corlocs
        self._metric_prefix = (metric_prefix + '_') if metric_prefix else ''

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing -
                standard_fields.InputDataFields.groundtruth_boxes: float32
                    numpy array of shape [num_boxes, 4] containing `num_boxes`
                    groundtruth boxes of the format [ymin, xmin, ymax, xmax] in
                    absolute image coordinates.
                standard_fields.InputDataFields.groundtruth_classes: integer
                    numpy array of shape [num_boxes] containing 1-indexed
                    groundtruth classes for the boxes.
                standard_fields.InputDataFields.groundtruth_instance_masks:
                    Optional numpy array of shape [num_boxes, height, width]
                    with values in {0, 1}.

        Raises:
            ValueError: On adding groundtruth for an image more than once. Will
                also raise error if instance masks are not in groundtruth
                dictionary.
        """
        if image_id in self._image_ids:
            raise ValueError(
                'Image with id {} already added.'.format(image_id))

        groundtruth_classes = (
            groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_classes] -
            self._label_id_offset)

        groundtruth_masks = None
        if self._evaluate_masks:
            if (standard_fields.InputDataFields.groundtruth_instance_masks
                    not in groundtruth_dict):
                raise ValueError(
                    'Instance masks not in groundtruth dictionary.')
            groundtruth_masks = groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_instance_masks]
        self._evaluation.add_single_ground_truth_image_info(
            image_key=image_id,
            groundtruth_boxes=groundtruth_dict[
                standard_fields.InputDataFields.groundtruth_boxes],
            groundtruth_class_labels=groundtruth_classes,
            groundtruth_masks=groundtruth_masks,
        )
        self._image_ids.update([image_id])

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary containing -
                standard_fields.DetectionResultFields.detection_boxes: float32
                    numpy array of shape [num_boxes, 4] containing `num_boxes`
                    detection boxes of the format [ymin, xmin, ymax, xmax] in
                    absolute image coordinates.
                standard_fields.DetectionResultFields.detection_scores: float32
                    numpy array of shape [num_boxes] containing detection
                    scores for the boxes.
                standard_fields.DetectionResultFields.detection_classes:
                    integer numpy array of shape [num_boxes] containing
                    1-indexed detection classes for the boxes.
                standard_fields.DetectionResultFields.detection_masks: uint8
                    numpy array of shape [num_boxes, height, width] containing
                    `num_boxes` masks of values ranging between 0 and 1.

        Raises:
            ValueError: If detection masks are not in detections dictionary.
        """
        detection_classes = (
            detections_dict[
                standard_fields.DetectionResultFields.detection_classes] -
            self._label_id_offset)
        detection_masks = None
        if self._evaluate_masks:
            if (standard_fields.DetectionResultFields.detection_masks
                    not in detections_dict):
                raise ValueError(
                    'Detection masks not in detections dictionary.')
            detection_masks = detections_dict[
                standard_fields.DetectionResultFields.detection_masks]
        self._evaluation.add_single_detected_image_info(
            image_key=image_id,
            detected_boxes=detections_dict[
                standard_fields.DetectionResultFields.detection_boxes],
            detected_scores=detections_dict[
                standard_fields.DetectionResultFields.detection_scores],
            detected_class_labels=detection_classes,
            detected_masks=detection_masks,
        )

    @staticmethod
    def create_category_index(categories):
        """Creates dictionary of COCO compatible categories keyed by category
        id.

        Args:
            categories: a list of dicts, each of which has the following keys:
                'id': (required) an integer id uniquely identifying this
                    category.
                'name': (required) string representing category name
                    e.g., 'cat', 'dog', 'pizza'.

        Returns:
            category_index: a dict containing the same entries as categories,
                but keyed by the 'id' field of each category.
        """
        category_index = {}
        for cat in categories:
            category_index[cat['id']] = cat
        return category_index

    def evaluate(self):
        """Compute evaluation result.

        Returns:
            A dictionary of metrics with the following fields -

            1. summary_metrics:
                'Precision/mAP@<matching_iou_threshold>IOU': mean average
                precision at the specified IOU threshold

            2. per_category_ap: category specific results with keys of the form
               'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'
        """
        (per_class_ap, mean_ap, _, _, per_class_corloc,
         mean_corloc) = self._evaluation.evaluate()

        metric = f'mAP@{self._matching_iou_threshold}IOU'
        pascal_metrics = {self._metric_prefix + metric: mean_ap}
        if self._evaluate_corlocs:
            pascal_metrics[self._metric_prefix +
                           'Precision/meanCorLoc@{}IOU'.format(
                               self._matching_iou_threshold)] = mean_corloc
        category_index = self.create_category_index(self._categories)
        for idx in range(per_class_ap.size):
            if idx + self._label_id_offset in category_index:
                display_name = (
                    self._metric_prefix +
                    'PerformanceByCategory/AP@{}IOU/{}'.format(
                        self._matching_iou_threshold,
                        category_index[idx + self._label_id_offset]['name'],
                    ))
                pascal_metrics[display_name] = per_class_ap[idx]

                # Optionally add CorLoc metrics.classes
                if self._evaluate_corlocs:
                    display_name = (
                        self._metric_prefix +
                        'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                            self._matching_iou_threshold,
                            category_index[idx +
                                           self._label_id_offset]['name'],
                        ))
                    pascal_metrics[display_name] = per_class_corloc[idx]

        return pascal_metrics

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._evaluation = ObjectDetectionEvaluation(
            num_groundtruth_classes=self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset,
        )
        self._image_ids.clear()


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluate detections using PASCAL metrics."""

    def __init__(self, categories, matching_iou_threshold=0.5):
        super(PascalDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            evaluate_corlocs=False,
            use_weighted_mean_ap=False,
        )


ObjectDetectionEvalMetrics = collections.namedtuple(
    'ObjectDetectionEvalMetrics',
    [
        'average_precisions',
        'mean_ap',
        'precisions',
        'recalls',
        'corlocs',
        'mean_corloc',
    ],
)


class ObjectDetectionEvaluation:
    """Internal implementation of Pascal object detection metrics."""

    def __init__(self,
                 num_groundtruth_classes,
                 matching_iou_threshold=0.5,
                 nms_iou_threshold=1.0,
                 nms_max_output_boxes=10000,
                 use_weighted_mean_ap=False,
                 label_id_offset=0):
        if num_groundtruth_classes < 1:
            raise ValueError(
                'Need at least 1 groundtruth class for evaluation.')

        self.per_image_eval = per_image_evaluation.PerImageEvaluation(
            num_groundtruth_classes=num_groundtruth_classes,
            matching_iou_threshold=matching_iou_threshold,
        )
        self.num_class = num_groundtruth_classes
        self.use_weighted_mean_ap = use_weighted_mean_ap
        self.label_id_offset = label_id_offset

        self.groundtruth_boxes = {}
        self.groundtruth_class_labels = {}
        self.groundtruth_masks = {}
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=int)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

        self._initialize_detections()

    def _initialize_detections(self):
        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(
            self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = []
        self.recalls_per_class = []
        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def clear_detections(self):
        self._initialize_detections()

    def add_single_ground_truth_image_info(self,
                                           image_key,
                                           groundtruth_boxes,
                                           groundtruth_class_labels,
                                           groundtruth_masks=None):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
            image_key: A unique string/integer identifier for the image.
            groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
                containing `num_boxes` groundtruth boxes of the format
                [ymin, xmin, ymax, xmax] in absolute image coordinates.
            groundtruth_class_labels: integer numpy array of shape [num_boxes]
                containing 0-indexed groundtruth classes for the boxes.
            groundtruth_masks: uint8 numpy array of shape
                [num_boxes, height, width] containing `num_boxes` groundtruth
                masks. The mask values range from 0 to 1.
        """
        if image_key in self.groundtruth_boxes:
            warnings.warn(('image %s has already been added to the ground '
                           'truth database.'), image_key)
            return

        self.groundtruth_boxes[image_key] = groundtruth_boxes
        self.groundtruth_class_labels[image_key] = groundtruth_class_labels
        self.groundtruth_masks[image_key] = groundtruth_masks

        self._update_ground_truth_statistics(groundtruth_class_labels)

    def add_single_detected_image_info(self,
                                       image_key,
                                       detected_boxes,
                                       detected_scores,
                                       detected_class_labels,
                                       detected_masks=None):
        """Adds detections for a single image to be used for evaluation.

        Args:
            image_key: A unique string/integer identifier for the image.
            detected_boxes: float32 numpy array of shape [num_boxes, 4]
                containing `num_boxes` detection boxes of the format
                [ymin, xmin, ymax, xmax] in absolute image coordinates.
            detected_scores: float32 numpy array of shape [num_boxes]
                containing detection scores for the boxes.
            detected_class_labels: integer numpy array of shape [num_boxes]
                containing 0-indexed detection classes for the boxes.
            detected_masks: np.uint8 numpy array of shape
                [num_boxes, height, width] containing `num_boxes` detection
                masks with values ranging between 0 and 1.

        Raises:
            ValueError: if the number of boxes, scores and class labels differ
                in length.
        """
        if len(detected_boxes) != len(detected_scores) or len(
                detected_boxes) != len(detected_class_labels):
            raise ValueError(
                'detected_boxes, detected_scores and '
                'detected_class_labels should all have same lengths. Got'
                '[%d, %d, %d]' % len(detected_boxes),
                len(detected_scores),
                len(detected_class_labels),
            )

        if image_key in self.detection_keys:
            warnings.warn(('image %s has already been added to the ground '
                           'truth database.'), image_key)
            return

        self.detection_keys.add(image_key)
        if image_key in self.groundtruth_boxes:
            groundtruth_boxes = self.groundtruth_boxes[image_key]
            groundtruth_class_labels = self.groundtruth_class_labels[image_key]
            # Masks are popped instead of look up. The reason is that we do not
            # want to keep all masks in memory which can cause memory overflow.
            groundtruth_masks = self.groundtruth_masks.pop(image_key)
        else:
            groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
            groundtruth_class_labels = np.array([], dtype=int)
            if detected_masks is None:
                groundtruth_masks = None
            else:
                groundtruth_masks = np.empty(shape=[0, 1, 1], dtype=float)
        (
            scores,
            tp_fp_labels,
        ) = self.per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_class_labels,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_class_labels=groundtruth_class_labels,
            detected_masks=detected_masks,
            groundtruth_masks=groundtruth_masks,
        )

        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])

    def _update_ground_truth_statistics(self, groundtruth_class_labels):
        """Update grouth truth statitistics.

        Args:
            groundtruth_class_labels: An integer numpy array of length M,
                representing M class labels of object instances in ground truth
        """
        count = defaultdict(lambda: 0)
        for label in groundtruth_class_labels:
            count[label] += 1
        for k in count:
            self.num_gt_instances_per_class[k] += count[k]
            self.num_gt_imgs_per_class[k] += 1

    def evaluate(self):
        """Compute evaluation result.

        Returns:
            A named tuple with the following fields -
                average_precision: float numpy array of average precision for
                    each class.
                mean_ap: mean average precision of all classes, float scalar
                precisions: List of precisions, each precision is a float numpy
                    array
                recalls: List of recalls, each recall is a float numpy array
                corloc: numpy float array
                mean_corloc: Mean CorLoc score for each class, float scalar
        """
        if (self.num_gt_instances_per_class == 0).any():
            logging.info(
                'The following classes have no ground truth examples: %s',
                np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) +
                self.label_id_offset)

        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)

        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=bool)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(
                    self.tp_fp_labels_per_class[class_index])
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = metrics.compute_precision_recall(
                scores, tp_fp_labels,
                self.num_gt_instances_per_class[class_index])
            self.precisions_per_class.append(precision)
            self.recalls_per_class.append(recall)
            average_precision = metrics.compute_average_precision(
                precision, recall)
            self.average_precision_per_class[class_index] = average_precision

        self.corloc_per_class = metrics.compute_cor_loc(
            self.num_gt_imgs_per_class,
            self.num_images_correctly_detected_per_class)

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = metrics.compute_precision_recall(
                all_scores, all_tp_fp_labels, num_gt_instances)
            mean_ap = metrics.compute_average_precision(precision, recall)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        mean_corloc = np.nanmean(self.corloc_per_class)
        return ObjectDetectionEvalMetrics(
            self.average_precision_per_class,
            mean_ap,
            self.precisions_per_class,
            self.recalls_per_class,
            self.corloc_per_class,
            mean_corloc,
        )
