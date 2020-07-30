import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .ohem_hinge_loss import OHEMHingeLoss


@LOSSES.register_module()
class SSNLoss(nn.Module):

    def activity_loss(self, activity_score, labels, activity_indexer):
        """Activity Loss.

        It will calculate activity loss given activity_score and label.

        Args：
            activity_score (torch.Tensor): Predicted activity score.
            labels (torch.Tensor): Groundtruth class label.
            activity_indexer (torch.Tensor): Index slices of proposals.

        Returns:
            torch.Tensor: Returned cross entropy loss.
        """
        pred = activity_score[activity_indexer, :]
        gt = labels[activity_indexer]
        return F.cross_entropy(pred, gt)

    def completeness_loss(self,
                          completeness_score,
                          labels,
                          completeness_indexer,
                          positive_per_video,
                          incomplete_per_video,
                          ohem_ratio=0.17):
        """Completeness Loss.

        It will calculate completeness loss given completeness_score and label.

        Args：
            completeness_score (torch.Tensor): Predicted completeness score.
            labels (torch.Tensor): Groundtruth class label.
            completeness_indexer (torch.Tensor): Index slices of positive and
                incomplete proposals.
            positive_per_video (int): Number of positive proposals sampled
                per video.
            incomplete_per_video (int): Number of incomplete proposals sampled
                pre video.
            ohem_ratio (float): Ratio of online hard example mining.
                Default: 0.17.

        Returns:
            torch.Tensor: Returned class-wise completeness loss.
        """
        pred = completeness_score[completeness_indexer, :]
        gt = labels[completeness_indexer]

        pred_dim = pred.size(1)
        pred = pred.view(-1, positive_per_video + incomplete_per_video,
                         pred_dim)
        gt = gt.view(-1, positive_per_video + incomplete_per_video)

        # yapf:disable
        positive_pred = pred[:, :positive_per_video, :].contiguous().view(-1, pred_dim)  # noqa:E501
        incomplete_pred = pred[:, positive_per_video:, :].contiguous().view(-1, pred_dim)  # noqa:E501
        # yapf:enable

        positive_loss = OHEMHingeLoss.apply(
            positive_pred, gt[:, :positive_per_video].contiguous().view(-1), 1,
            1.0, positive_per_video)
        incomplete_loss = OHEMHingeLoss.apply(
            incomplete_pred, gt[:, positive_per_video:].contiguous().view(-1),
            -1, ohem_ratio, incomplete_per_video)
        num_positives = positive_pred.size(0)
        num_incompletes = int(incomplete_pred.size(0) * ohem_ratio)

        return ((positive_loss + incomplete_loss) /
                float(num_positives + num_incompletes))

    def classwise_regression_loss(self, bbox_pred, labels, bbox_targets,
                                  regression_indexer):
        """Classwise Regression Loss.

        It will calculate classwise_regression loss given
        class_reg_pred and targets.

        Args：
            bbox_pred (torch.Tensor): Predicted interval center and span
                of positive proposals.
            labels (torch.Tensor): Groundtruth class label.
            bbox_targets (torch.Tensor): Groundtruth center and span
                of positive proposals.
            regression_indexer (torch.Tensor): Index slices of
                positive proposals.

        Returns:
            torch.Tensor: Returned class-wise regression loss.
        """
        pred = bbox_pred[regression_indexer, :, :]
        gt = labels[regression_indexer]
        reg_target = bbox_targets[regression_indexer, :]

        class_idx = gt.data - 1
        classwise_pred = pred[:, class_idx, :]
        classwise_reg_pred = torch.cat(
            (torch.diag(classwise_pred[:, :, 0]).view(
                -1, 1), torch.diag(classwise_pred[:, :, 1]).view(-1, 1)),
            dim=1)
        loss = F.smooth_l1_loss(
            classwise_reg_pred.view(-1), reg_target.view(-1)) * 2
        return loss

    def forward(self, activity_score, completeness_score, bbox_pred,
                proposal_type, labels, bbox_targets, train_cfg):
        """Calculate Boundary Matching Network Loss.

        Args:
            activity_score (torch.Tensor): Predicted activity score.
            completeness_score (torch.Tensor): Predicted completeness score.
            bbox_pred (torch.Tensor): Predicted interval center and span
                of positive proposals.
            proposal_type (torch.Tensor): Type index slices of proposals.
            labels (torch.Tensor): Groundtruth class label.
            bbox_targets (torch.Tensor): Groundtruth center and span
                of positive proposals.
            train_cfg (dict): Config for training.

        Returns:
            dict([torch.Tensor, torch.Tensor, torch.Tensor]):
                (loss_activity, loss_completeness, loss_reg).
                Loss_activity is the activity loss, loss_completeness is
                the class-wise completeness loss,
                loss_reg is the class-wise regression loss.
        """
        self.sampler = train_cfg.ssn.sampler
        self.loss_weight = train_cfg.ssn.loss_weight
        losses = dict()

        proposal_type = proposal_type.view(-1)
        labels = labels.view(-1)
        activity_indexer = ((proposal_type == 0) +
                            (proposal_type == 2)).nonzero().squeeze(1)
        completeness_indexer = ((proposal_type == 0) +
                                (proposal_type == 1)).nonzero().squeeze(1)

        total_ratio = (
            self.sampler.positive_ratio + self.sampler.background_ratio +
            self.sampler.incomplete_ratio)
        positive_per_video = int(self.sampler.num_per_video *
                                 (self.sampler.positive_ratio / total_ratio))
        background_per_video = int(
            self.sampler.num_per_video *
            (self.sampler.background_ratio / total_ratio))
        incomplete_per_video = (
            self.sampler.num_per_video - positive_per_video -
            background_per_video)

        losses['loss_activity'] = self.activity_loss(activity_score, labels,
                                                     activity_indexer)

        losses['loss_completeness'] = self.completeness_loss(
            completeness_score,
            labels,
            completeness_indexer,
            positive_per_video,
            incomplete_per_video,
            ohem_ratio=positive_per_video / incomplete_per_video)
        losses['loss_completeness'] *= self.loss_weight.comp_loss_weight

        if bbox_pred is not None:
            regression_indexer = (proposal_type == 0).nonzero().squeeze(1)
            bbox_targets = bbox_targets.view(-1, 2)
            losses['loss_reg'] = self.classwise_regression_loss(
                bbox_pred, labels, bbox_targets, regression_indexer)
            losses['loss_reg'] *= self.loss_weight.reg_loss_weight

        return losses
