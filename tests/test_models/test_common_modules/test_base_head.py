import torch
import torch.nn.functional as F
from mmcv.utils import assert_dict_has_keys

from mmaction.models import BaseHead


class ExampleHead(BaseHead):
    # use an ExampleHead to test BaseHead
    def init_weights(self):
        pass

    def forward(self, x):
        pass


def test_base_head():
    head = ExampleHead(3, 400, dict(type='CrossEntropyLoss'))

    cls_scores = torch.rand((3, 4))
    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2] * 3).squeeze()
    losses = head.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'

    head = ExampleHead(3, 400, dict(type='CrossEntropyLoss', loss_weight=2.0))

    cls_scores = torch.rand((3, 4))
    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2] * 3).squeeze()
    losses = head.loss(cls_scores, gt_labels)
    assert_dict_has_keys(losses, ['loss_cls'])
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'

    # Test Soft label with batch size > 1
    cls_scores = torch.rand((3, 3))
    gt_labels = torch.LongTensor([[2] * 3])
    gt_one_hot_labels = F.one_hot(gt_labels, num_classes=3).squeeze()
    losses = head.loss(cls_scores, gt_one_hot_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'

    # Test Soft label with batch size = 1
    cls_scores = torch.rand((1, 3))
    gt_labels = torch.LongTensor([2])
    gt_one_hot_labels = F.one_hot(gt_labels, num_classes=3).squeeze()
    losses = head.loss(cls_scores, gt_one_hot_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'

    # test multi-class & label smoothing
    head = ExampleHead(
        3,
        400,
        dict(type='BCELossWithLogits'),
        multi_class=True,
        label_smooth_eps=0.1)

    # batch size > 1
    cls_scores = torch.rand((2, 3))
    gt_labels = torch.LongTensor([[1, 0, 1], [0, 1, 0]]).squeeze()
    losses = head.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'

    # batch size = 1
    cls_scores = torch.rand((1, 3))
    gt_labels = torch.LongTensor([[1, 0, 1]]).squeeze()
    losses = head.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'
