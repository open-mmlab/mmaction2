import torch

from mmaction.models import BaseHead


class ExampleHead(BaseHead):
    # use a ExampleHead to success BaseHead
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
