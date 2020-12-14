import torch

from ..registry import HEADS
from .tsn_head import TSNHead


@HEADS.register_module()
class NullHead(TSNHead):

    def __init__(self, num_classes, in_channels):
        super().__init__(num_classes=num_classes, in_channels=in_channels)

    def forward(self, x, num_segs=None):
        # [N * num_segs, in_channels]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels]
        x = self.consensus(x)
        # [N, 1, in_channels]
        x = x.squeeze(1)
        # [N, in_channels]
        return x

    def init_weights(self):
        pass

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        # import pdb
        # pdb.set_trace()
        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
