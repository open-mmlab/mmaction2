# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.datasets import CutmixBlending, MixupBlending


def test_mixup():
    alpha = 0.2
    num_classes = 10
    label = torch.randint(0, num_classes, (4, ))
    mixup = MixupBlending(num_classes, alpha)

    # NCHW imgs
    imgs = torch.randn(4, 4, 3, 32, 32)
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 3, 32, 32))
    assert mixed_label.shape == torch.Size((4, num_classes))

    # NCTHW imgs
    imgs = torch.randn(4, 4, 2, 3, 32, 32)
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 2, 3, 32, 32))
    assert mixed_label.shape == torch.Size((4, num_classes))


def test_cutmix():
    alpha = 0.2
    num_classes = 10
    label = torch.randint(0, num_classes, (4, ))
    mixup = CutmixBlending(num_classes, alpha)

    # NCHW imgs
    imgs = torch.randn(4, 4, 3, 32, 32)
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 3, 32, 32))
    assert mixed_label.shape == torch.Size((4, num_classes))

    # NCTHW imgs
    imgs = torch.randn(4, 4, 2, 3, 32, 32)
    mixed_imgs, mixed_label = mixup(imgs, label)
    assert mixed_imgs.shape == torch.Size((4, 4, 2, 3, 32, 32))
    assert mixed_label.shape == torch.Size((4, num_classes))
