# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .binary_logistic_regression_loss import binary_logistic_regression_loss


@LOSSES.register_module()
class BMNLoss(nn.Module):
    """BMN Loss.

    From paper https://arxiv.org/abs/1907.09702,
    code https://github.com/JJBOY/BMN-Boundary-Matching-Network.
    It will calculate loss for BMN Model. This loss is a weighted sum of

        1) temporal evaluation loss based on confidence score of start and
        end positions.
        2) proposal evaluation regression loss based on confidence scores of
        candidate proposals.
        3) proposal evaluation classification loss based on classification
        results of candidate proposals.
    """

    @staticmethod
    def tem_loss(pred_start, pred_end, gt_start, gt_end):
        """Calculate Temporal Evaluation Module Loss.

        This function calculate the binary_logistic_regression_loss for start
        and end respectively and returns the sum of their losses.

        Args:
            pred_start (torch.Tensor): Predicted start score by BMN model.
            pred_end (torch.Tensor): Predicted end score by BMN model.
            gt_start (torch.Tensor): Groundtruth confidence score for start.
            gt_end (torch.Tensor): Groundtruth confidence score for end.

        Returns:
            torch.Tensor: Returned binary logistic loss.
        """
        loss_start = binary_logistic_regression_loss(pred_start, gt_start)
        loss_end = binary_logistic_regression_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    @staticmethod
    def pem_reg_loss(pred_score,
                     gt_iou_map,
                     mask,
                     high_temporal_iou_threshold=0.7,
                     low_temporal_iou_threshold=0.3):
        """Calculate Proposal Evaluation Module Regression Loss.

        Args:
            pred_score (torch.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (torch.Tensor): Groundtruth temporal_iou score.
            mask (torch.Tensor): Boundary-Matching mask.
            high_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.7.
            low_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.3.

        Returns:
            torch.Tensor: Proposal evaluation regression loss.
        """
        u_hmask = (gt_iou_map > high_temporal_iou_threshold).float()
        u_mmask = ((gt_iou_map <= high_temporal_iou_threshold) &
                   (gt_iou_map > low_temporal_iou_threshold)).float()
        u_lmask = ((gt_iou_map <= low_temporal_iou_threshold) &
                   (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.rand_like(gt_iou_map)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.rand_like(gt_iou_map)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(
            loss * torch.ones_like(weights)) / torch.sum(weights)

        return loss

    @staticmethod
    def pem_cls_loss(pred_score,
                     gt_iou_map,
                     mask,
                     threshold=0.9,
                     ratio_range=(1.05, 21),
                     eps=1e-5):
        """Calculate Proposal Evaluation Module Classification Loss.

        Args:
            pred_score (torch.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (torch.Tensor): Groundtruth temporal_iou score.
            mask (torch.Tensor): Boundary-Matching mask.
            threshold (float): Threshold of temporal_iou for positive
                instances. Default: 0.9.
            ratio_range (tuple): Lower bound and upper bound for ratio.
                Default: (1.05, 21)
            eps (float): Epsilon for small value. Default: 1e-5

        Returns:
            torch.Tensor: Proposal evaluation classification loss.
        """
        pmask = (gt_iou_map > threshold).float()
        nmask = (gt_iou_map <= threshold).float()
        nmask = nmask * mask

        num_positive = max(torch.sum(pmask), 1)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / num_positive
        ratio = torch.clamp(ratio, ratio_range[0], ratio_range[1])

        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio

        loss_pos = coef_1 * torch.log(pred_score + eps) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + eps) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss

    def forward(self,
                pred_bm,
                pred_start,
                pred_end,
                gt_iou_map,
                gt_start,
                gt_end,
                bm_mask,
                weight_tem=1.0,
                weight_pem_reg=10.0,
                weight_pem_cls=1.0):
        """Calculate Boundary Matching Network Loss.

        Args:
            pred_bm (torch.Tensor): Predicted confidence score for boundary
                matching map.
            pred_start (torch.Tensor): Predicted confidence score for start.
            pred_end (torch.Tensor): Predicted confidence score for end.
            gt_iou_map (torch.Tensor): Groundtruth score for boundary matching
                map.
            gt_start (torch.Tensor): Groundtruth temporal_iou score for start.
            gt_end (torch.Tensor): Groundtruth temporal_iou score for end.
            bm_mask (torch.Tensor): Boundary-Matching mask.
            weight_tem (float): Weight for tem loss. Default: 1.0.
            weight_pem_reg (float): Weight for pem regression loss.
                Default: 10.0.
            weight_pem_cls (float): Weight for pem classification loss.
                Default: 1.0.

        Returns:
            tuple([torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                (loss, tem_loss, pem_reg_loss, pem_cls_loss). Loss is the bmn
                loss, tem_loss is the temporal evaluation loss, pem_reg_loss is
                the proposal evaluation regression loss, pem_cls_loss is the
                proposal evaluation classification loss.
        """
        pred_bm_reg = pred_bm[:, 0].contiguous()
        pred_bm_cls = pred_bm[:, 1].contiguous()
        gt_iou_map = gt_iou_map * bm_mask

        pem_reg_loss = self.pem_reg_loss(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss(pred_bm_cls, gt_iou_map, bm_mask)
        tem_loss = self.tem_loss(pred_start, pred_end, gt_start, gt_end)
        loss = (
            weight_tem * tem_loss + weight_pem_reg * pem_reg_loss +
            weight_pem_cls * pem_cls_loss)
        return loss, tem_loss, pem_reg_loss, pem_cls_loss
