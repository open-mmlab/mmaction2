# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.model.weight_init import constant_init, kaiming_init

from mmaction.registry import MODELS
from .utils import post_processing, temporal_iop


@MODELS.register_module()
class TEM(BaseModel):
    """Temporal Evaluation Model for Boundary Sensitive Network.

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
        super().__init__()

        self.temporal_dim = temporal_dim
        self.boundary_ratio = boundary_ratio
        self.feat_dim = tem_feat_dim
        self.c_hidden = tem_hidden_dim
        self.match_threshold = tem_match_threshold
        self.output_dim = output_dim
        self.loss_cls = MODELS.build(loss_cls)
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

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _temporal_anchors(self, tmin_offset=0., tmax_offset=1.):
        """Generate temporal anchors.

        Args:
            tmin_offset (int): Offset for the minimum value of temporal anchor.
                Default: 0.
            tmax_offset (int): Offset for the maximum value of temporal anchor.
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

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            dict: A dictionary of loss components.
        """
        tem_output = self._forward(batch_inputs)

        score_action = tem_output[:, 0, :]
        score_start = tem_output[:, 1, :]
        score_end = tem_output[:, 2, :]

        gt_bbox = [
            sample.gt_instances['gt_bbox'] for sample in batch_data_samples
        ]
        label_action, label_start, label_end = self.generate_labels(gt_bbox)
        device = batch_inputs.device
        label_action = label_action.to(device)
        label_start = label_start.to(device)
        label_end = label_end.to(device)

        loss_action = self.loss_cls(score_action, label_action,
                                    self.match_threshold)
        loss_start = self.loss_cls(score_start, label_start,
                                   self.match_threshold)
        loss_end = self.loss_cls(score_end, label_end, self.match_threshold)

        loss_dict = {
            'loss_action': loss_action * self.loss_weight,
            'loss_start': loss_start,
            'loss_end': loss_end
        }

        return loss_dict

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        """Define the computation performed at every call when testing."""
        tem_output = self._forward(batch_inputs).cpu().numpy()
        batch_action = tem_output[:, 0, :]
        batch_start = tem_output[:, 1, :]
        batch_end = tem_output[:, 2, :]

        video_results = []
        for batch_idx, _ in enumerate(batch_action):
            video_name = batch_data_samples[batch_idx].metainfo['video_name']
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
        # TODO: do this without numpy
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
        if type(inputs) is not torch.Tensor:
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


@MODELS.register_module()
class PEM(BaseModel):
    """Proposals Evaluation Model for Boundary Sensitive Network.

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
                 pem_feat_dim: int,
                 pem_hidden_dim: int,
                 pem_u_ratio_m: float,
                 pem_u_ratio_l: float,
                 pem_high_temporal_iou_threshold: float,
                 pem_low_temporal_iou_threshold: float,
                 soft_nms_alpha: float,
                 soft_nms_low_threshold: float,
                 soft_nms_high_threshold: float,
                 post_process_top_k: int,
                 feature_extraction_interval: int = 16,
                 fc1_ratio: float = 0.1,
                 fc2_ratio: float = 0.1,
                 output_dim: int = 1):
        super().__init__()

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

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = F.relu(self.fc1_ratio * self.fc1(x))
        x = torch.sigmoid(self.fc2_ratio * self.fc2(x))
        return x

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            dict: A dictionary of loss components.
        """
        device = self.fc1.weight.device

        bsp_feature = torch.cat([
            sample.gt_instances['bsp_feature'] for sample in batch_data_samples
        ]).to(device)

        reference_temporal_iou = torch.cat([
            sample.gt_instances['reference_temporal_iou']
            for sample in batch_data_samples
        ]).to(device)

        pem_output = self._forward(bsp_feature)

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

    def _parse(self, gt_instances, key):
        out = torch.cat([gt[key] for gt in gt_instances])
        out = out.view(-1).cpu().numpy().reshape(-1, 1)
        return out

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        """Define the computation performed at every call when testing."""
        device = self.fc1.weight.device

        bsp_feature = torch.cat([
            sample.gt_instances['bsp_feature'] for sample in batch_data_samples
        ]).to(device)

        pem_output = self._forward(bsp_feature).view(-1).cpu().numpy()
        pem_output = pem_output.reshape(-1, 1)

        gt_instances = [sample.gt_instances for sample in batch_data_samples]

        tmin = self._parse(gt_instances, 'tmin')
        tmax = self._parse(gt_instances, 'tmax')
        tmin_score = self._parse(gt_instances, 'tmin_score')
        tmax_score = self._parse(gt_instances, 'tmax_score')

        score = np.array(pem_output * tmin_score * tmax_score).reshape(-1, 1)
        result = np.concatenate(
            (tmin, tmax, tmin_score, tmax_score, pem_output, score), axis=1)
        result = result.reshape(-1, 6)

        video_info = batch_data_samples[0].metainfo
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
            batch_inputs (Tensor): The input tensor with shape
                (N, C, ...) in general.
            batch_data_samples (List[:obj:`ActionDataSample`], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
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
