# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmaction.structures import bbox2result


def test_bbox2result():
    bboxes = torch.tensor([[0.072, 0.47, 0.84, 0.898],
                           [0.23, 0.215, 0.781, 0.534],
                           [0.195, 0.128, 0.643, 0.944],
                           [0.236, 0.189, 0.689, 0.74],
                           [0.375, 0.371, 0.726, 0.804],
                           [0.024, 0.398, 0.776, 0.719]])
    labels = torch.tensor([[-1.650, 0.515, 0.798, 1.240],
                           [1.368, -1.128, 0.037, -1.087],
                           [0.481, -1.303, 0.501, -0.463],
                           [-0.356, 0.126, -0.840, 0.438],
                           [0.079, 1.269, -0.263, -0.538],
                           [-0.853, 0.391, 0.103, 0.398]])
    num_classes = 4
    #  Test for multi-label
    result = bbox2result(bboxes, labels, num_classes)
    assert np.all(
        np.isclose(
            result[0],
            np.array([[0.072, 0.47, 0.84, 0.898, 0.515],
                      [0.236, 0.189, 0.689, 0.74, 0.126],
                      [0.375, 0.371, 0.726, 0.804, 1.269],
                      [0.024, 0.398, 0.776, 0.719, 0.391]])))
    assert np.all(
        np.isclose(
            result[1],
            np.array([[0.072, 0.47, 0.84, 0.898, 0.798],
                      [0.23, 0.215, 0.781, 0.534, 0.037],
                      [0.195, 0.128, 0.643, 0.944, 0.501],
                      [0.024, 0.398, 0.776, 0.719, 0.103]])))
    assert np.all(
        np.isclose(
            result[2],
            np.array([[0.072, 0.47, 0.84, 0.898, 1.24],
                      [0.236, 0.189, 0.689, 0.74, 0.438],
                      [0.024, 0.398, 0.776, 0.719, 0.398]])))

    # Test for single-label
    result = bbox2result(bboxes, labels, num_classes, -1.0)
    assert np.all(
        np.isclose(result[0], np.array([[0.375, 0.371, 0.726, 0.804, 1.269]])))
    assert np.all(
        np.isclose(
            result[1],
            np.array([[0.23, 0.215, 0.781, 0.534, 0.037],
                      [0.195, 0.128, 0.643, 0.944, 0.501]])))
    assert np.all(
        np.isclose(
            result[2],
            np.array([[0.072, 0.47, 0.84, 0.898, 1.240],
                      [0.236, 0.189, 0.689, 0.74, 0.438],
                      [0.024, 0.398, 0.776, 0.719, 0.398]])))
