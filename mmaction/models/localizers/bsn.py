import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...localization import temporal_iop
from ..builder import LOCALIZERS, build_loss
from .base import BaseLocalizer
from .utils import post_processing


@LOCALIZERS.register_module()
class TEM(BaseLocalizer):
    """Temporal Evaluation Model for Boundary Sensetive Network.

    Please refer `BSN: Boundary Sensitive Network for Temporal Action
    Proposal Generation <http://arxiv.org/abs/1806.02964>`_.

    Code reference
    https://github.com/wzmsltw/BSN-boundary-sensitive-network

    Args:
        tem_feat_dim (int): Feature dimension.
        tem_hidden_dim (int): Hidden layer dimension.
        tem_match_threshold (float): Temporal evaluation match threshold.
        loss_cls (dict): Config for building loss.
            Default: ``dict(type='BinaryLogisticRegressionLoss')``.
        loss_weight (float): Weight term for action_loss. Default: 2.
        output_dim (int): Output dimension. Default: 3.
        conv1_ratio (float): Ratio of conv1 layer output. Default: 1.0.
        conv2_ratio (float): Ratio of conv2 layer output. Default: 1.0.
        conv3_ratio (float): Ratio of conv3 layer output. Default: 0.01.
    """

    def __init__(self,
                 temporal_dim,
                 boundary_ratio,
                 tem_feat_dim,
                 tem_hidden_dim,
                 tem_match_threshold,
                 loss_cls=dict(type='BinaryLogisticRegressionLoss'),
                 loss_weight=2,
                 output_dim=3,
                 conv1_ratio=1,
                 conv2_ratio=1,
                 conv3_ratio=0.01):
        super(BaseLocalizer, self).__init__()

        self.temporal_dim = temporal_dim
        self.boundary_ratio = boundary_ratio
        self.feat_dim = tem_feat_dim
        self.c_hidden = tem_hidden_dim
        self.match_threshold = tem_match_threshold
        self.output_dim = output_dim
        self.loss_cls = build_loss(loss_cls)
        self.loss_weight = loss_weight
        self.conv1_ratio = conv1_ratio
        self.conv2_ratio = conv2_ratio
        self.conv3_ratio = conv3_ratio

        self.conv1 = nn.Conv1d(
            in_channels=self.feat_dim,
            out_channels=self.c_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1)
        self.conv2 = nn.Conv1d(
            in_channels=self.c_hidden,
            out_channels=self.c_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1)
        self.conv3 = nn.Conv1d(
            in_channels=self.c_hidden,
            out_channels=self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0)
        self.anchors_tmins, self.anchors_tmaxs = self._temporal_anchors()

    def _temporal_anchors(self, tmin_offset=0., tmax_offset=1.):
        """Generate temporal anchors.

        Args:
            tmin_offset (int): Offset for the minimum value of temporal anchor.
                Default: 0.
            tmax_offset (int): Offset for the maximun value of temporal anchor.
                Default: 1.

        Returns:
            tuple[Sequence[float]]: The minimum and maximum values of temporal
                anchors.
        """
        temporal_gap = 1. / self.temporal_dim
        anchors_tmins = []
        anchors_tmaxs = []
        for i in range(self.temporal_dim):
            anchors_tmins.append(temporal_gap * (i + tmin_offset))
            anchors_tmaxs.append(temporal_gap * (i + tmax_offset))

        return anchors_tmins, anchors_tmaxs

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = F.relu(self.conv1_ratio * self.conv1(x))
        x = F.relu(self.conv2_ratio * self.conv2(x))
        x = torch.sigmoid(self.conv3_ratio * self.conv3(x))
        return x

    def forward_train(self, raw_feature, label_action, label_start, label_end):
        """Define the computation performed at every call when training."""
        tem_output = self._forward(raw_feature)
        score_action = tem_output[:, 0, :]
        score_start = tem_output[:, 1, :]
        score_end = tem_output[:, 2, :]

        loss_action = self.loss_cls(score_action, label_action,
                                    self.match_threshold)
        loss_start_small = self.loss_cls(score_start, label_start,
                                         self.match_threshold)
        loss_end_small = self.loss_cls(score_end, label_end,
                                       self.match_threshold)
        loss_dict = {
            'loss_action': loss_action * self.loss_weight,
            'loss_start': loss_start_small,
            'loss_end': loss_end_small
        }

        return loss_dict

    def forward_test(self, raw_feature, video_meta):
        """Define the computation performed at every call when testing."""
        tem_output = self._forward(raw_feature).cpu().numpy()
        batch_action = tem_output[:, 0, :]
        batch_start = tem_output[:, 1, :]
        batch_end = tem_output[:, 2, :]

        video_meta_list = [dict(x) for x in video_meta]

        video_results = []

        for batch_idx, _ in enumerate(batch_action):
            video_name = video_meta_list[batch_idx]['video_name']
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]
            video_result = np.stack((video_action, video_start, video_end,
                                     self.anchors_tmins, self.anchors_tmaxs),
                                    axis=1)
            video_results.append((video_name, video_result))
        return video_results

    def generate_labels(self, gt_bbox):
        """Generate training labels."""
        match_score_action_list = []
        match_score_start_list = []
        match_score_end_list = []
        for every_gt_bbox in gt_bbox:
            gt_tmins = every_gt_bbox[:, 0].cpu().numpy()
            gt_tmaxs = every_gt_bbox[:, 1].cpu().numpy()

            gt_lens = gt_tmaxs - gt_tmins
            gt_len_pad = np.maximum(1. / self.temporal_dim,
                                    self.boundary_ratio * gt_lens)

            gt_start_bboxs = np.stack(
                (gt_tmins - gt_len_pad / 2, gt_tmins + gt_len_pad / 2), axis=1)
            gt_end_bboxs = np.stack(
                (gt_tmaxs - gt_len_pad / 2, gt_tmaxs + gt_len_pad / 2), axis=1)

            match_score_action = []
            match_score_start = []
            match_score_end = []

            for anchor_tmin, anchor_tmax in zip(self.anchors_tmins,
                                                self.anchors_tmaxs):
                match_score_action.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax, gt_tmins,
                                     gt_tmaxs)))
                match_score_start.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax,
                                     gt_start_bboxs[:, 0], gt_start_bboxs[:,
                                                                          1])))
                match_score_end.append(
                    np.max(
                        temporal_iop(anchor_tmin, anchor_tmax,
                                     gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
            match_score_action_list.append(match_score_action)
            match_score_start_list.append(match_score_start)
            match_score_end_list.append(match_score_end)
        match_score_action_list = torch.Tensor(match_score_action_list)
        match_score_start_list = torch.Tensor(match_score_start_list)
        match_score_end_list = torch.Tensor(match_score_end_list)
        return (match_score_action_list, match_score_start_list,
                match_score_end_list)

    def forward(self,
                raw_feature,
                gt_bbox=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            label_action, label_start, label_end = (
                self.generate_labels(gt_bbox))
            device = raw_feature.device
            label_action = label_action.to(device)
            label_start = label_start.to(device)
            label_end = label_end.to(device)
            return self.forward_train(raw_feature, label_action, label_start,
                                      label_end)

        return self.forward_test(raw_feature, video_meta)


@LOCALIZERS.register_module()
class PEM(BaseLocalizer):
    """Proposals Evaluation Model for Boundary Sensetive Network.

    Please refer `BSN: Boundary Sensitive Network for Temporal Action
    Proposal Generation <http://arxiv.org/abs/1806.02964>`_.

    Code reference
    https://github.com/wzmsltw/BSN-boundary-sensitive-network

    Args:
        pem_feat_dim (int): Feature dimension.
        pem_hidden_dim (int): Hidden layer dimension.
        pem_u_ratio_m (float): Ratio for medium score proprosals to balance
            data.
        pem_u_ratio_l (float): Ratio for low score proprosals to balance data.
        pem_high_temporal_iou_threshold (float): High IoU threshold.
        pem_low_temporal_iou_threshold (float): Low IoU threshold.
        soft_nms_alpha (float): Soft NMS alpha.
        soft_nms_low_threshold (float): Soft NMS low threshold.
        soft_nms_high_threshold (float): Soft NMS high threshold.
        post_process_top_k (int): Top k proposals in post process.
        feature_extraction_interval (int):
            Interval used in feature extraction. Default: 16.
        fc1_ratio (float): Ratio for fc1 layer output. Default: 0.1.
        fc2_ratio (float): Ratio for fc2 layer output. Default: 0.1.
        output_dim (int): Output dimension. Default: 1.
    """

    def __init__(self,
                 pem_feat_dim,
                 pem_hidden_dim,
                 pem_u_ratio_m,
                 pem_u_ratio_l,
                 pem_high_temporal_iou_threshold,
                 pem_low_temporal_iou_threshold,
                 soft_nms_alpha,
                 soft_nms_low_threshold,
                 soft_nms_high_threshold,
                 post_process_top_k,
                 feature_extraction_interval=16,
                 fc1_ratio=0.1,
                 fc2_ratio=0.1,
                 output_dim=1):
        super(BaseLocalizer, self).__init__()

        self.feat_dim = pem_feat_dim
        self.hidden_dim = pem_hidden_dim
        self.u_ratio_m = pem_u_ratio_m
        self.u_ratio_l = pem_u_ratio_l
        self.pem_high_temporal_iou_threshold = pem_high_temporal_iou_threshold
        self.pem_low_temporal_iou_threshold = pem_low_temporal_iou_threshold
        self.soft_nms_alpha = soft_nms_alpha
        self.soft_nms_low_threshold = soft_nms_low_threshold
        self.soft_nms_high_threshold = soft_nms_high_threshold
        self.post_process_top_k = post_process_top_k
        self.feature_extraction_interval = feature_extraction_interval
        self.fc1_ratio = fc1_ratio
        self.fc2_ratio = fc2_ratio
        self.output_dim = output_dim

        self.fc1 = nn.Linear(
            in_features=self.feat_dim, out_features=self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            bias=True)

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = torch.cat(list(x))
        x = F.relu(self.fc1_ratio * self.fc1(x))
        x = torch.sigmoid(self.fc2_ratio * self.fc2(x))
        return x

    def forward_train(self, bsp_feature, reference_temporal_iou):
        """Define the computation performed at every call when training."""
        pem_output = self._forward(bsp_feature)
        reference_temporal_iou = torch.cat(list(reference_temporal_iou))
        device = pem_output.device
        reference_temporal_iou = reference_temporal_iou.to(device)

        anchors_temporal_iou = pem_output.view(-1)
        u_hmask = (reference_temporal_iou >
                   self.pem_high_temporal_iou_threshold).float()
        u_mmask = (
            (reference_temporal_iou <= self.pem_high_temporal_iou_threshold)
            & (reference_temporal_iou > self.pem_low_temporal_iou_threshold)
        ).float()
        u_lmask = (reference_temporal_iou <=
                   self.pem_low_temporal_iou_threshold).float()

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = self.u_ratio_m * num_h / (num_m)
        r_m = torch.min(r_m, torch.Tensor([1.0]).to(device))[0]
        u_smmask = torch.rand(u_hmask.size()[0], device=device)
        u_smmask = u_smmask * u_mmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = self.u_ratio_l * num_h / (num_l)
        r_l = torch.min(r_l, torch.Tensor([1.0]).to(device))[0]
        u_slmask = torch.rand(u_hmask.size()[0], device=device)
        u_slmask = u_slmask * u_lmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        temporal_iou_weights = u_hmask + u_smmask + u_slmask
        temporal_iou_loss = F.smooth_l1_loss(anchors_temporal_iou,
                                             reference_temporal_iou)
        temporal_iou_loss = torch.sum(
            temporal_iou_loss *
            temporal_iou_weights) / torch.sum(temporal_iou_weights)
        loss_dict = dict(temporal_iou_loss=temporal_iou_loss)

        return loss_dict

    def forward_test(self, bsp_feature, tmin, tmax, tmin_score, tmax_score,
                     video_meta):
        """Define the computation performed at every call when testing."""
        pem_output = self._forward(bsp_feature).view(-1).cpu().numpy().reshape(
            -1, 1)

        tmin = tmin.view(-1).cpu().numpy().reshape(-1, 1)
        tmax = tmax.view(-1).cpu().numpy().reshape(-1, 1)
        tmin_score = tmin_score.view(-1).cpu().numpy().reshape(-1, 1)
        tmax_score = tmax_score.view(-1).cpu().numpy().reshape(-1, 1)
        score = np.array(pem_output * tmin_score * tmax_score).reshape(-1, 1)
        result = np.concatenate(
            (tmin, tmax, tmin_score, tmax_score, pem_output, score), axis=1)
        result = result.reshape(-1, 6)
        video_info = dict(video_meta[0])
        proposal_list = post_processing(result, video_info,
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

    def forward(self,
                bsp_feature,
                reference_temporal_iou=None,
                tmin=None,
                tmax=None,
                tmin_score=None,
                tmax_score=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(bsp_feature, reference_temporal_iou)

        return self.forward_test(bsp_feature, tmin, tmax, tmin_score,
                                 tmax_score, video_meta)
