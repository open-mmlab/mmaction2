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
"""Contains classes specifying naming conventions used for object detection.

Specifies:
  InputDataFields: standard fields used by reader/preprocessor/batcher.
  DetectionResultFields: standard fields returned by object detector.
"""


class InputDataFields:
    """Names for the input tensors.

    Holds the standard data field names to use for identifying input tensors.
    This should be used by the decoder to identify keys for the returned
    tensor_dict containing input tensors. And it should be used by the model to
    identify the tensors it needs.

    Attributes:
        image: image.
        original_image: image in the original input size.
        key: unique key corresponding to image.
        source_id: source of the original image.
        filename: original filename of the dataset (without common path).
        groundtruth_image_classes: image-level class labels.
        groundtruth_boxes: coordinates of the ground truth boxes in the image.
        groundtruth_classes: box-level class labels.
        groundtruth_label_types: box-level label types (e.g. explicit
            negative).
        groundtruth_is_crowd: [DEPRECATED, use groundtruth_group_of instead]
            is the groundtruth a single object or a crowd.
        groundtruth_area: area of a groundtruth segment.
        groundtruth_difficult: is a `difficult` object
        groundtruth_group_of: is a `group_of` objects, e.g. multiple objects of
            the same class, forming a connected group, where instances are
            heavily occluding each other.
        proposal_boxes: coordinates of object proposal boxes.
        proposal_objectness: objectness score of each proposal.
        groundtruth_instance_masks: ground truth instance masks.
        groundtruth_instance_boundaries: ground truth instance boundaries.
        groundtruth_instance_classes: instance mask-level class labels.
        groundtruth_keypoints: ground truth keypoints.
        groundtruth_keypoint_visibilities: ground truth keypoint visibilities.
        groundtruth_label_scores: groundtruth label scores.
        groundtruth_weights: groundtruth weight factor for bounding boxes.
        num_groundtruth_boxes: number of groundtruth boxes.
        true_image_shapes: true shapes of images in the resized images, as
            resized images can be padded with zeros.
    """

    image = 'image'
    original_image = 'original_image'
    key = 'key'
    source_id = 'source_id'
    filename = 'filename'
    groundtruth_image_classes = 'groundtruth_image_classes'
    groundtruth_boxes = 'groundtruth_boxes'
    groundtruth_classes = 'groundtruth_classes'
    groundtruth_label_types = 'groundtruth_label_types'
    groundtruth_is_crowd = 'groundtruth_is_crowd'
    groundtruth_area = 'groundtruth_area'
    groundtruth_difficult = 'groundtruth_difficult'
    groundtruth_group_of = 'groundtruth_group_of'
    proposal_boxes = 'proposal_boxes'
    proposal_objectness = 'proposal_objectness'
    groundtruth_instance_masks = 'groundtruth_instance_masks'
    groundtruth_instance_boundaries = 'groundtruth_instance_boundaries'
    groundtruth_instance_classes = 'groundtruth_instance_classes'
    groundtruth_keypoints = 'groundtruth_keypoints'
    groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
    groundtruth_label_scores = 'groundtruth_label_scores'
    groundtruth_weights = 'groundtruth_weights'
    num_groundtruth_boxes = 'num_groundtruth_boxes'
    true_image_shape = 'true_image_shape'


class DetectionResultFields:
    """Naming conventions for storing the output of the detector.

    Attributes:
        source_id: source of the original image.
        key: unique key corresponding to image.
        detection_boxes: coordinates of the detection boxes in the image.
        detection_scores: detection scores for the detection boxes in the
            image.
        detection_classes: detection-level class labels.
        detection_masks: contains a segmentation mask for each detection box.
        detection_boundaries: contains an object boundary for each detection
            box.
        detection_keypoints: contains detection keypoints for each detection
            box.
        num_detections: number of detections in the batch.
    """

    source_id = 'source_id'
    key = 'key'
    detection_boxes = 'detection_boxes'
    detection_scores = 'detection_scores'
    detection_classes = 'detection_classes'
    detection_masks = 'detection_masks'
    detection_boundaries = 'detection_boundaries'
    detection_keypoints = 'detection_keypoints'
    num_detections = 'num_detections'
