# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

# from mmaction.utils import register_all_modules
from mmaction.registry import MODELS
from .utils import temporal_iop


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
        # register_all_modules()

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
        pass

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
        tem_output = self._forward(batch_inputs)

        score_action = tem_output[:, 0, :]
        score_start = tem_output[:, 1, :]
        score_end = tem_output[:, 2, :]

        gt_bbox = [sample['gt_bbox'] for sample in batch_data_samples]
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
            video_name = batch_data_samples[batch_idx]['video_name']
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

    def forward(self, batch_inputs, batch_data_samples, mode, **kwargs):
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
        if mode == 'tensor':
            return self._forward(batch_inputs, **kwargs)
        if mode == 'predict':
            return self.predict(batch_inputs, batch_data_samples, **kwargs)
        elif mode == 'loss':
            return self.loss(batch_inputs, batch_data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
