# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModel
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import OptConfigType
from .utils import (batch_iou, bbox_se_transform_batch, bbox_se_transform_inv,
                    bbox_xw_transform_batch, bbox_xw_transform_inv,
                    post_processing)


class LGTE(BaseModel):
    """Local-Global Temporal Encoder (LGTE)

    Args:
        input_dim (int): Input feature dimension.
        dropout (float): the dropout rate for the residual branch of
            self-attention and ffn.
        temporal_dim (int): Total frames selected for each video.
            Defaults to 100.
        window_size (int): the window size for Local Temporal Encoder.
            Defaults to 9.
        init_cfg (dict or ConfigDict, optional): The Config for
            initialization. Defaults to None.
    """

    def __init__(self,
                 input_dim: int,
                 dropout: float,
                 temporal_dim: int = 100,
                 window_size: int = 9,
                 num_heads: int = 8,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(LGTE, self).__init__(init_cfg)

        self.atten = MultiheadAttention(
            embed_dims=input_dim,
            num_heads=num_heads,
            proj_drop=dropout,
            attn_drop=0.1)
        self.ffn = FFN(
            embed_dims=input_dim, feedforward_channels=256, ffn_drop=dropout)

        norm_cfg = dict(type='LN', eps=1e-6)
        self.norm1 = build_norm_layer(norm_cfg, input_dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, input_dim)[1]

        mask = self._mask_matrix(num_heads, temporal_dim, window_size)
        self.register_buffer('mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        """Forward call for LGTE.

        Args:
            x (torch.Tensor): The input tensor with shape (B, C, L)
        """
        x = x.permute(2, 0, 1)
        mask = self.mask.repeat(x.size(1), 1, 1, 1)
        L = x.shape[0]
        x = self.atten(x, attn_mask=mask.reshape(-1, L, L))
        x = self.norm1(x)
        x = self.ffn(x)
        x = self.norm2(x)
        x = x.permute(1, 2, 0)
        return x

    @staticmethod
    def _mask_matrix(num_heads: int, temporal_dim: int,
                     window_size: int) -> Tensor:
        mask = torch.zeros(num_heads, temporal_dim, temporal_dim)
        index = torch.arange(temporal_dim)

        for i in range(num_heads // 2):
            for j in range(temporal_dim):
                ignored = (index - j).abs() > window_size / 2
                mask[i, j] = ignored

        return mask.unsqueeze(0).bool()


def StartEndRegressor(sample_num: int, feat_dim: int) -> nn.Module:
    """Start and End Regressor in the Temporal Boundary Regressor.

    Args:
        sample_num (int): number of samples for the start & end.
        feat_dim (int): feature dimension.

    Returns:
        A pytorch module that works as the start and end regressor. The input
        of the module should have a shape of (B, feat_dim * 2, sample_num).
    """
    hidden_dim = 128
    regressor = nn.Sequential(
        nn.Conv1d(
            feat_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=8,
            stride=2), nn.ReLU(inplace=True),
        nn.Conv1d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=8,
            stride=2), nn.ReLU(inplace=True),
        nn.Conv1d(hidden_dim * 2, 2, kernel_size=sample_num // 4, groups=2),
        nn.Flatten())
    return regressor


def CenterWidthRegressor(temporal_len: int, feat_dim: int) -> nn.Module:
    """Center Width in the Temporal Boundary Regressor.

    Args:
        temporal_len (int): temporal dimension of the inputs.
        feat_dim (int): feature dimension.

    Returns:
        A pytorch module that works as the start and end regressor. The input
        of the module should have a shape of (B, feat_dim, temporal_len).
    """
    hidden_dim = 512
    regressor = nn.Sequential(
        nn.Conv1d(
            feat_dim, hidden_dim, kernel_size=3, padding=1, groups=4,
            stride=2), nn.ReLU(inplace=True),
        nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=4,
            stride=2), nn.ReLU(inplace=True),
        nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=temporal_len // 4, groups=4),
        nn.ReLU(inplace=True), nn.Conv1d(hidden_dim, 3, kernel_size=1))
    return regressor


class TemporalTransform:
    """Temporal Transform to sample temporal features."""

    def __init__(self, prop_boundary_ratio: float, action_sample_num: int,
                 se_sample_num: int, temporal_interval: int):
        super(TemporalTransform, self).__init__()
        self.temporal_interval = temporal_interval
        self.prop_boundary_ratio = prop_boundary_ratio
        self.action_sample_num = action_sample_num
        self.se_sample_num = se_sample_num

    def __call__(self, segments: Tensor, features: Tensor) -> List[Tensor]:
        s_len = segments[:, 1] - segments[:, 0]
        starts_segments = [
            segments[:, 0] - self.prop_boundary_ratio * s_len, segments[:, 0]
        ]
        starts_segments = torch.stack(starts_segments, dim=1)

        ends_segments = [
            segments[:, 1], segments[:, 1] + self.prop_boundary_ratio * s_len
        ]
        ends_segments = torch.stack(ends_segments, dim=1)

        starts_feature = self._sample_one_temporal(starts_segments,
                                                   self.se_sample_num,
                                                   features)
        ends_feature = self._sample_one_temporal(ends_segments,
                                                 self.se_sample_num, features)
        actions_feature = self._sample_one_temporal(segments,
                                                    self.action_sample_num,
                                                    features)
        return starts_feature, actions_feature, ends_feature

    def _sample_one_temporal(self, segments: Tensor, out_len: int,
                             features: Tensor) -> Tensor:
        segments = segments.clamp(0, 1) * 2 - 1
        theta = segments.new_zeros((features.size(0), 2, 3))
        theta[:, 1, 1] = 1.0
        theta[:, 0, 0] = (segments[:, 1] - segments[:, 0]) / 2.0
        theta[:, 0, 2] = (segments[:, 1] + segments[:, 0]) / 2.0

        size = torch.Size((*features.shape[:2], 1, out_len))
        grid = F.affine_grid(theta, size)
        stn_feature = F.grid_sample(features.unsqueeze(2), grid)
        stn_feature = stn_feature.view(*features.shape[:2], out_len)
        return stn_feature


class TBR(BaseModel):
    """Temporal Boundary Regressor (TBR)"""

    def __init__(self,
                 se_sample_num: int,
                 action_sample_num: int,
                 temporal_dim: int,
                 prop_boundary_ratio: float = 0.5,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(TBR, self).__init__(init_cfg)

        hidden_dim = 512

        self.reg1se = StartEndRegressor(se_sample_num, hidden_dim)
        temporal_len = se_sample_num * 2 + action_sample_num
        self.reg1xw = CenterWidthRegressor(temporal_len, hidden_dim)
        self.ttn = TemporalTransform(prop_boundary_ratio, action_sample_num,
                                     se_sample_num, temporal_dim)

    def forward(self, proposals: Tensor, features: Tensor, gt_boxes: Tensor,
                iou_thres: float, training: bool) -> tuple:
        proposals1 = proposals[:, :2]
        starts_feat1, actions_feat1, ends_feat1 = self.ttn(
            proposals1, features)

        reg1se = self.reg1se(torch.cat([starts_feat1, ends_feat1], dim=1))

        features1xw = torch.cat([starts_feat1, actions_feat1, ends_feat1],
                                dim=2)
        reg1xw = self.reg1xw(features1xw).squeeze(2)

        preds_iou1 = reg1xw[:, 2].sigmoid()
        reg1xw = reg1xw[:, :2]

        if training:
            proposals2xw = bbox_xw_transform_inv(proposals1, reg1xw, 0.1, 0.2)
            proposals2se = bbox_se_transform_inv(proposals1, reg1se, 1.0)

            iou1 = batch_iou(proposals1, gt_boxes)
            targets1se = bbox_se_transform_batch(proposals1, gt_boxes)
            targets1xw = bbox_xw_transform_batch(proposals1, gt_boxes)
            rloss1se = self.regress_loss(reg1se, targets1se, iou1, iou_thres)
            rloss1xw = self.regress_loss(reg1xw, targets1xw, iou1, iou_thres)
            rloss1 = rloss1se + rloss1xw
            iloss1 = self.iou_loss(preds_iou1, iou1, iou_thres=iou_thres)
        else:
            proposals2xw = bbox_xw_transform_inv(proposals1, reg1xw, 0.1, 0.2)
            proposals2se = bbox_se_transform_inv(proposals1, reg1se, 0.2)
            rloss1 = iloss1 = 0
        proposals2 = (proposals2se + proposals2xw) / 2.0
        proposals2 = torch.clamp(proposals2, min=0.)
        return preds_iou1, proposals2, rloss1, iloss1

    def regress_loss(self, regression, targets, iou_with_gt, iou_thres):
        weight = (iou_with_gt >= iou_thres).float().unsqueeze(1)
        reg_loss = F.smooth_l1_loss(regression, targets, reduction='none')
        if weight.sum() > 0:
            reg_loss = (weight * reg_loss).sum() / weight.sum()
        else:
            reg_loss = (weight * reg_loss).sum()
        return reg_loss

    def iou_loss(self, preds_iou, match_iou, iou_thres):
        preds_iou = preds_iou.view(-1)
        u_hmask = (match_iou > iou_thres).float()
        u_mmask = ((match_iou <= iou_thres) & (match_iou > 0.3)).float()
        u_lmask = (match_iou <= 0.3).float()

        num_h, num_m, num_l = u_hmask.sum(), u_mmask.sum(), u_lmask.sum()

        bs, device = u_hmask.size()[0], u_hmask.device

        r_m = min(num_h / num_m, 1)
        u_smmask = torch.rand(bs, device=device) * u_mmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = min(num_h / num_l, 1)
        u_slmask = torch.rand(bs, device=device) * u_lmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        iou_weights = u_hmask + u_smmask + u_slmask
        iou_loss = F.smooth_l1_loss(preds_iou, match_iou, reduction='none')
        if iou_weights.sum() > 0:
            iou_loss = (iou_loss * iou_weights).sum() / iou_weights.sum()
        else:
            iou_loss = (iou_loss * iou_weights).sum()
        return iou_loss


@MODELS.register_module()
class TCANet(BaseModel):
    """Temporal Context Aggregation Network.

    Please refer `Temporal Context Aggregation Network for Temporal Action
    Proposal Refinement <https://arxiv.org/abs/2103.13141>`_.
    Code Reference:
    https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch
    """

    def __init__(self,
                 feat_dim: int = 2304,
                 se_sample_num: int = 32,
                 action_sample_num: int = 64,
                 temporal_dim: int = 100,
                 window_size: int = 9,
                 lgte_num: int = 2,
                 soft_nms_alpha: float = 0.4,
                 soft_nms_low_threshold: float = 0.0,
                 soft_nms_high_threshold: float = 0.0,
                 post_process_top_k: int = 100,
                 feature_extraction_interval: int = 16,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(TCANet, self).__init__(init_cfg)

        self.soft_nms_alpha = soft_nms_alpha
        self.soft_nms_low_threshold = soft_nms_low_threshold
        self.soft_nms_high_threshold = soft_nms_high_threshold
        self.feature_extraction_interval = feature_extraction_interval
        self.post_process_top_k = post_process_top_k

        hidden_dim = 512
        self.x_1d_b_f = nn.Sequential(
            nn.Conv1d(
                feat_dim, hidden_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
        )

        for i in 1, 2, 3:
            tbr = TBR(
                se_sample_num=se_sample_num,
                action_sample_num=action_sample_num,
                temporal_dim=temporal_dim,
                init_cfg=init_cfg,
                **kwargs)
            setattr(self, f'tbr{i}', tbr)

        self.lgtes = nn.ModuleList([
            LGTE(
                input_dim=hidden_dim,
                dropout=0.1,
                temporal_dim=temporal_dim,
                window_size=window_size,
                init_cfg=init_cfg,
                **kwargs) for i in range(lgte_num)
        ])

    def forward(self, inputs, data_samples, mode, **kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[:obj:`ActionDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if not isinstance(input, Tensor):
            inputs = torch.stack(inputs)
        if mode == 'tensor':
            return self._forward(inputs, **kwargs)
        if mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.x_1d_b_f(x)
        for layer in self.lgtes:
            x = layer(x)
        return x

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        features = self._forward(batch_inputs)
        proposals_ = [
            sample.proposals['proposals'] for sample in batch_data_samples
        ]

        batch_size = len(proposals_)
        proposals_num = max([_.shape[0] for _ in proposals_])

        proposals = torch.zeros((batch_size, proposals_num, 3),
                                device=features.device)
        for i, proposal in enumerate(proposals_):
            proposals[i, :proposal.shape[0]] = proposal

        gt_boxes_ = [
            sample.gt_instances['gt_bbox'] for sample in batch_data_samples
        ]
        gt_boxes = torch.zeros((batch_size, proposals_num, 2),
                               device=features.device)
        for i, gt_box in enumerate(gt_boxes_):
            L = gt_box.shape[0]
            if L <= proposals_num:
                gt_boxes[i, :L] = gt_box
            else:
                random_index = torch.randperm(L)[:proposals_num]
                gt_boxes[i] = gt_box[random_index]

        for i in range(batch_size):
            proposals[i, :, 2] = i
        proposals = proposals.view(batch_size * proposals_num, 3)
        proposals_select = proposals[:, 0:2].sum(dim=1) > 0
        proposals = proposals[proposals_select, :]

        features = features[proposals[:, 2].long()]

        gt_boxes = gt_boxes.view(batch_size * proposals_num, 2)
        gt_boxes = gt_boxes[proposals_select, :]

        _, proposals1, rloss1, iloss1 = self.tbr1(proposals, features,
                                                  gt_boxes, 0.5, True)
        _, proposals2, rloss2, iloss2 = self.tbr2(proposals1, features,
                                                  gt_boxes, 0.6, True)
        _, _, rloss3, iloss3 = self.tbr3(proposals2, features, gt_boxes, 0.7,
                                         True)

        loss_dict = dict(
            rloss1=rloss1,
            rloss2=rloss2,
            rloss3=rloss3,
            iloss1=iloss1,
            iloss2=iloss2,
            iloss3=iloss3)
        return loss_dict

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        features = self._forward(batch_inputs)
        proposals_ = [
            sample.proposals['proposals'] for sample in batch_data_samples
        ]

        batch_size = len(proposals_)
        proposals_num = max([_.shape[0] for _ in proposals_])

        proposals = torch.zeros((batch_size, proposals_num, 3),
                                device=features.device)
        for i, proposal in enumerate(proposals_):
            proposals[i, :proposal.shape[0]] = proposal

        scores = proposals[:, :, 2]
        for i in range(batch_size):
            proposals[i, :, 2] = i

        proposals = proposals.view(batch_size * proposals_num, 3)
        proposals_select = proposals[:, 0:2].sum(dim=1) > 0
        proposals = proposals[proposals_select, :]
        scores = scores.view(-1)[proposals_select]

        features = features[proposals[:, 2].long()]

        preds_iou1, proposals1 = self.tbr1(proposals, features, None, 0.5,
                                           False)[:2]
        preds_iou2, proposals2 = self.tbr2(proposals1, features, None, 0.6,
                                           False)[:2]
        preds_iou3, proposals3 = self.tbr3(proposals2, features, None, 0.7,
                                           False)[:2]

        all_proposals = []
        # all_proposals = [proposals]
        all_proposals += [
            torch.cat([proposals1, (scores * preds_iou1).view(-1, 1)], dim=1)
        ]
        all_proposals += [
            torch.cat([proposals2, (scores * preds_iou2).view(-1, 1)], dim=1)
        ]
        all_proposals += [
            torch.cat([proposals3, (scores * preds_iou3).view(-1, 1)], dim=1)
        ]

        all_proposals = torch.cat(all_proposals, dim=0).cpu().numpy()
        video_info = batch_data_samples[0].metainfo
        proposal_list = post_processing(all_proposals, video_info,
                                        self.soft_nms_alpha,
                                        self.soft_nms_low_threshold,
                                        self.soft_nms_high_threshold,
                                        self.post_process_top_k,
                                        self.feature_extraction_interval)
        output = [
            dict(
                video_name=video_info['video_name'],
                proposal_list=proposal_list)
        ]
        return output
