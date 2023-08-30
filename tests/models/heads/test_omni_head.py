# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmaction.models import OmniHead


class obj():

    def __init__(self, name, value):
        super(obj, self).__init__()
        setattr(self, name, value)


def testOmniHead():
    head = OmniHead(image_classes=100, video_classes=200, in_channels=400)

    image_feat = torch.randn(2, 400, 8, 8)
    image_score = head(image_feat)
    assert image_score.shape == torch.Size([2, 100])

    video_feat = torch.randn(2, 400, 8, 8, 8)
    video_score = head(video_feat)
    assert video_score.shape == torch.Size([2, 200])

    head = OmniHead(
        image_classes=100,
        video_classes=200,
        in_channels=400,
        video_nl_head=True)

    video_feat = torch.randn(2, 400, 8, 8, 8)
    video_score = head(video_feat)
    assert video_score.shape == torch.Size([2, 200])
    data_samples = [obj('gt_label', torch.tensor(1)) for _ in range(2)]
    losses = head.loss_by_feat(video_score, data_samples)
    assert 'loss_cls' in losses

    image_feat = torch.randn(1, 400, 8, 8)
    head.eval()
    image_score = head(image_feat)
    assert image_score.shape == torch.Size([1, 100])
    data_samples = [obj('gt_label', torch.tensor(1))]
    losses = head.loss_by_feat(image_score, data_samples)
    assert 'loss_cls' in losses
