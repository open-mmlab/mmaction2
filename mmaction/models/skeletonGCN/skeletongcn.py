from ..builder import RECOGNIZERS
from .base import BaseGCN


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""

    def forward_train(self, skeletons, labels, **kwargs):
        assert self.with_cls_head
        losses = dict()

        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)
        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        return losses

    def forward_test(self, skeletons):
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output
