# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import LabelData

from mmaction.models import CutmixBlending, MixupBlending
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
