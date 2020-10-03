import torch  # isort: skip
import torch.nn.functional as F  # isort: skip

from ..registry import LOSSES  # isort: skip
from .base import BaseWeightedLoss  # isort: skip


@LOSSES.register_module()
class HVULoss(BaseWeightedLoss):
    """Calculate the BCELoss for HVU.

    Args:
        categories (list[str]): Names of tag categories, tags are organized in
            this order. Default: ['action', 'attribute', 'concept', 'event',
            'object', 'scene'].
        category_nums (list[int]): Number of tags for each category. Default:
            [739, 117, 291, 69, 1679, 248].
        category_loss_weights (list[float]): Loss weights of categories, it
            applies only if `loss_type == 'individual'`. The loss weights will
            be normalized so that the sum equals to 1, so that you can give any
            positive number as loss weight. Default: [1, 1, 1, 1, 1, 1].
        loss_type (str): The loss type we calculate, we can either calculate
            the BCELoss for all tags, or calculate the BCELoss for tags in each
            category. Choices are ['individual', 'all']. Default: 'all'.
        with_mask (bool): Since some tag categories are missing for some video
            clips. If `with_mask == True`, we will not calculate loss for these
            missing categories. Otherwise, these missing categories are treated
            as negative samples.
        loss_weight (float): The loss weight. Default: 1.0.
    """

    def __init__(self,
                 categories=[
                     'action', 'attribute', 'concept', 'event', 'object',
                     'scene'
                 ],
                 category_nums=[739, 117, 291, 69, 1679, 248],
                 category_loss_weights=[1, 1, 1, 1, 1, 1],
                 loss_type='all',
                 with_mask=False,
                 loss_weight=1.0):

        super().__init__(loss_weight)
        self.categories = categories
        self.category_nums = category_nums
        self.category_loss_weights = category_loss_weights
        for loss_weight in self.category_loss_weights:
            assert loss_weight >= 0
        self.loss_type = loss_type
        self.with_mask = with_mask
        self.category_startidx = [0]
        for i in range(len(self.category_nums) - 1):
            self.category_startidx.append(self.category_startidx[-1] +
                                          self.category_nums[i])
        assert self.loss_type in ['individual', 'all']

    def _forward(self, cls_score, label, aux_info):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            aux_info (dict[torch.Tensor]): Other auxiliary tensors needed for
                loss calculation. For HVU Loss, `aux_info` should contain
                `mask` and `category_mask`

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        assert 'mask' in aux_info
        assert 'category_mask' in aux_info
        mask = aux_info['mask']
        category_mask = aux_info['category_mask']

        if self.loss_type == 'all':
            loss_cls = F.binary_cross_entropy_with_logits(
                cls_score, label, reduction='none')
            if self.with_mask:
                w_loss_cls = mask * loss_cls
                w_loss_cls = torch.sum(
                    w_loss_cls, dim=1) / torch.sum(
                        mask, dim=1)
                w_loss_cls = torch.mean(w_loss_cls)
                return dict(loss_cls=w_loss_cls)
            else:
                return dict(loss_cls=torch.mean(loss_cls))
        elif self.loss_type == 'individual':
            losses = {}
            loss_weights = {}
            for name, num, start_idx in zip(self.categories,
                                            self.category_nums,
                                            self.category_startidx):
                category_score = cls_score[:, start_idx:start_idx + num]
                category_label = label[:, start_idx:start_idx + num]
                category_loss = F.binary_cross_entropy_with_logits(
                    category_score, category_label, reduction='none')
                category_loss = torch.mean(category_loss, dim=1)

                idx = self.categories.index(name)
                if self.with_mask:
                    category_mask = category_mask[:, idx].reshape(-1)
                    # there should be at least one sample which contains tags
                    # in thie category
                    if torch.sum(category_mask) < 0.5:
                        continue

                    category_loss = torch.sum(category_loss * category_mask)
                    category_loss = category_loss / torch.sum(category_mask)
                else:
                    category_loss = torch.mean(category_loss)
                # We name the loss of each category as 'LOSS', since we only
                # want to monitor them, not backward them. We will also provide
                # the loss used for backward in the losses dictionary
                losses[f'{name}_LOSS'] = category_loss
                loss_weights[f'{name}_LOSS'] = self.category_loss_weights[idx]
            loss_weight_sum = sum(loss_weights.values())
            loss_weights = {
                k: v / loss_weight_sum
                for k, v in loss_weights.items()
            }
            loss_cls = sum([losses[k] * loss_weights[k] for k in losses])
            losses['loss_cls'] = loss_cls
            # We also trace the loss weights
            losses.update({k + '_weight': v for k, v in loss_weights.item()})
            # Note that the loss weights are just for reference.
            return losses
