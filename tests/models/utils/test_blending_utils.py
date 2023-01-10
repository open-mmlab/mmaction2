# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmengine.structures import LabelData

from mmaction.models import CutmixBlending, MixupBlending, RandomBatchAugment
from mmaction.structures import ActionDataSample


def get_label(label_):
    label = []
    for idx, one_label in enumerate(label_):
        data_sample = ActionDataSample()
        data_sample.gt_labels = LabelData(item=label_[idx])
        label.append(data_sample)
    return label


def test_mixup():
    alpha = 0.2
    num_classes = 10
    label = get_label(torch.randint(0, num_classes, (4, )))
    mixup = MixupBlending(num_classes, alpha)

    # NCHW imgs
    imgs = torch.randn(4, 4, 3, 32, 32)
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 3, 32, 32))
    assert len(mixed_label) == 4

    # NCTHW imgs
    imgs = torch.randn(4, 4, 2, 3, 32, 32)
    label = get_label(torch.randint(0, num_classes, (4, )))
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 2, 3, 32, 32))
    assert len(mixed_label) == 4


def test_cutmix():
    alpha = 0.2
    num_classes = 10
    label = get_label(torch.randint(0, num_classes, (4, )))
    mixup = CutmixBlending(num_classes, alpha)

    # NCHW imgs
    imgs = torch.randn(4, 4, 3, 32, 32)
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 3, 32, 32))
    assert len(mixed_label) == 4

    # NCTHW imgs
    imgs = torch.randn(4, 4, 2, 3, 32, 32)
    label = get_label(torch.randint(0, num_classes, (4, )))
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 2, 3, 32, 32))
    assert len(mixed_label) == 4


def test_rand_blend():
    alpha_mixup = 0.2
    alpha_cutmix = 0.2
    num_classes = 10
    label = get_label(torch.randint(0, num_classes, (4, )))
    blending_augs = [
        dict(type='MixupBlending', alpha=alpha_mixup, num_classes=num_classes),
        dict(
            type='CutmixBlending', alpha=alpha_cutmix, num_classes=num_classes)
    ]

    # test assertion
    with pytest.raises(AssertionError):
        rand_mix = RandomBatchAugment(blending_augs, [0.5, 0.6])

    # mixup, cutmix
    rand_mix = RandomBatchAugment(blending_augs, probs=None)
    assert rand_mix.probs is None

    # mixup, cutmix and None
    probs = [0.5, 0.4]
    rand_mix = RandomBatchAugment(blending_augs, probs)

    np.testing.assert_allclose(rand_mix.probs[-1], 0.1)

    # test call
    imgs = torch.randn(4, 4, 3, 32, 32)  # NCHW imgs
    mixed_imgs, mixed_label = rand_mix(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 3, 32, 32))
    assert len(mixed_label) == 4

    imgs = torch.randn(4, 4, 2, 3, 32, 32)  # NCTHW imgs
    label = get_label(torch.randint(0, num_classes, (4, )))
    mixed_imgs, mixed_label = rand_mix(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 2, 3, 32, 32))
    assert len(mixed_label) == 4
