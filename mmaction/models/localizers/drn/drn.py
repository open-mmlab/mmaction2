# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mmaction.registry import MODELS
from mmaction.utils import OptConfigType
from ..utils import soft_nms
from .drn_utils import FPN, Backbone, FCOSModule, QueryEncoder


@MODELS.register_module()
class DRN(BaseModel):
    """Dense Regression Network for Video Grounding.

    Please refer `Dense Regression Network for Video Grounding
        <https://arxiv.org/abs/2103.13141>`_.
    Code Reference: https://github.com/Alvin-Zeng/DRN

    Args:
        vocab_size (int): number of all possible words in the query.
            Defaults to 1301.
        hidden_dim (int): the hidden dimension of the LSTM in the
            language model. Defaults to 512.
        embed_dim (int): the embedding dimension of the query. Defaults
            to 300.
        bidirection (bool): if True, use bi-direction LSTM in the
            language model. Defaults to True.
        first_output_dim (int): the output dimension of the first layer
            in the backbone. Defaults to 256.
        fpn_feature_dim (int): the output dimension of the FPN. Defaults
            to 512.
        feature_dim (int): the dimension of the video clip feature.
        lstm_layers (int): the number of LSTM layers in the language model.
            Defaults to 1.
        fcos_pre_nms_top_n (int): value of Top-N in the FCOS module before
            nms.  Defaults to 32.
        fcos_inference_thr (float): threshold in the FOCS inference. BBoxes
            with scores higher than this threshold are regarded as positive.
            Defaults to 0.05.
        fcos_prior_prob (float): A prior probability of the positive bboexes.
            Used to initialized the bias of the classification head.
            Defaults to 0.01.
        focal_alpha (float):Focal loss hyper-parameter alpha.
            Defaults to 0.25.
        focal_gamma (float): Focal loss hyper-parameter gamma.
            Defaults to 2.0.
        fpn_stride (Sequence[int]): the strides in the FPN. Defaults to
            [1, 2, 4].
        fcos_nms_thr (float): NMS threshold in the FOCS module.
            Defaults to 0.6.
        fcos_conv_layers (int): number of convolution layers in FCOS.
            Defaults to 1.
        fcos_num_class (int): number of classes in FCOS.
            Defaults to 2.
        is_first_stage (bool): if true, the model is in the first stage
            training.
        is_second_stage (bool): if true, the model is in the second stage
            training.
    """

    def __init__(self,
                 vocab_size: int = 1301,
                 hidden_dim: int = 512,
                 embed_dim: int = 300,
                 bidirection: bool = True,
                 first_output_dim: int = 256,
                 fpn_feature_dim: int = 512,
                 feature_dim: int = 4096,
                 lstm_layers: int = 1,
                 fcos_pre_nms_top_n: int = 32,
                 fcos_inference_thr: float = 0.05,
                 fcos_prior_prob: float = 0.01,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 fpn_stride: Sequence[int] = [1, 2, 4],
                 fcos_nms_thr: float = 0.6,
                 fcos_conv_layers: int = 1,
                 fcos_num_class: int = 2,
                 is_first_stage: bool = False,
                 is_second_stage: bool = False,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super(DRN, self).__init__(init_cfg)

        self.query_encoder = QueryEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=lstm_layers,
            bidirection=bidirection)

        channels_list = [
            (feature_dim + 256, first_output_dim, 3, 1),
            (first_output_dim, first_output_dim * 2, 3, 2),
            (first_output_dim * 2, first_output_dim * 4, 3, 2),
        ]
        self.backbone_net = Backbone(channels_list)

        self.fpn = FPN(
            in_channels_list=[256, 512, 1024], out_channels=fpn_feature_dim)

        self.fcos = FCOSModule(
            in_channels=fpn_feature_dim,
            fcos_num_class=fcos_num_class,
            fcos_conv_layers=fcos_conv_layers,
            fcos_prior_prob=fcos_prior_prob,
            fcos_inference_thr=fcos_inference_thr,
            fcos_pre_nms_top_n=fcos_pre_nms_top_n,
            fcos_nms_thr=fcos_nms_thr,
            test_detections_per_img=32,
            fpn_stride=fpn_stride,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            is_first_stage=is_first_stage,
            is_second_stage=is_second_stage)

        self.prop_fc = nn.Linear(feature_dim, feature_dim)
        self.position_transform = nn.Linear(3, 256)

        qInput = []
        for t in range(len(channels_list)):
            if t > 0:
                qInput += [nn.Linear(1024, channels_list[t - 1][1])]
            else:
                qInput += [nn.Linear(1024, feature_dim)]
        self.qInput = nn.ModuleList(qInput)

        self.is_second_stage = is_second_stage

    def forward(self, inputs, data_samples, mode, **kwargs):
        props_features = torch.stack(inputs)
        batch_size = props_features.shape[0]
        device = props_features.device
        proposals = torch.stack([
            sample.proposals['proposals'] for sample in data_samples
        ]).to(device)
        gt_bbox = torch.stack([
            sample.gt_instances['gt_bbox'] for sample in data_samples
        ]).to(device)

        video_info = [i.metainfo for i in data_samples]
        query_tokens_ = [i['query_tokens'] for i in video_info]
        query_length = [i['query_length'] for i in video_info]
        query_length = torch.from_numpy(np.array(query_length))

        max_query_len = max([i.shape[0] for i in query_tokens_])
        query_tokens = torch.zeros(batch_size, max_query_len)
        for idx, query_token in enumerate(query_tokens_):
            query_len = query_token.shape[0]
            query_tokens[idx, :query_len] = query_token

        query_tokens = query_tokens.to(device).long()
        query_length = query_length.to(device).long()  # should be on CPU!

        sort_index = query_length.argsort(descending=True)
        box_lists, loss_dict = self._forward(query_tokens[sort_index],
                                             query_length[sort_index],
                                             props_features[sort_index],
                                             proposals[sort_index],
                                             gt_bbox[sort_index])
        if mode == 'loss':
            return loss_dict
        elif mode == 'predict':
            # only support batch size = 1
            bbox = box_lists[0]

            per_vid_detections = bbox['detections']
            per_vid_scores = bbox['scores']

            props_pred = torch.cat(
                (per_vid_detections, per_vid_scores.unsqueeze(-1)), dim=-1)

            props_pred = props_pred.cpu().numpy()
            props_pred = sorted(props_pred, key=lambda x: x[-1], reverse=True)
            props_pred = np.array(props_pred)

            props_pred = soft_nms(
                props_pred,
                alpha=0.4,
                low_threshold=0.5,
                high_threshold=0.9,
                top_k=5)
            result = {
                'vid_name': data_samples[0].metainfo['vid_name'],
                'gt': gt_bbox[0].cpu().numpy(),
                'predictions': props_pred,
            }
            return [result]

        raise ValueError(f'Unsupported mode {mode}!')

    def nms_temporal(self, start, end, score, overlap=0.45):
        pick = []
        assert len(start) == len(score)
        assert len(end) == len(score)
        if len(start) == 0:
            return pick

        union = end - start
        # sort and get index
        intervals = [
            i[0] for i in sorted(enumerate(score), key=lambda x: x[1])
        ]

        while len(intervals) > 0:
            i = intervals[-1]
            pick.append(i)

            xx1 = [max(start[i], start[j]) for j in intervals[:-1]]
            xx2 = [min(end[i], end[j]) for j in intervals[:-1]]
            inter = [max(0., k2 - k1) for k1, k2 in zip(xx1, xx2)]
            o = [
                inter[u] / (union[i] + union[intervals[u]] - inter[u])
                for u in range(len(intervals) - 1)
            ]
            I_new = []
            for j in range(len(o)):
                if o[j] <= overlap:
                    I_new.append(intervals[j])
            intervals = I_new
        return np.array(pick)

    def _forward(self, query_tokens, query_length, props_features,
                 props_start_end, gt_bbox):

        position_info = [props_start_end, props_start_end]
        position_feats = []
        query_features = self.query_encoder(query_tokens, query_length)
        for i in range(len(query_features)):
            query_features[i] = self.qInput[i](query_features[i])
            if i > 1:
                position_info.append(
                    torch.cat([
                        props_start_end[:, ::2 * (i - 1), [0]],
                        props_start_end[:, 1::2 * (i - 1), [1]]
                    ],
                              dim=-1))
            props_duration = position_info[i][:, :, 1] - position_info[i][:, :,
                                                                          0]
            props_duration = props_duration.unsqueeze(-1)
            position_feat = torch.cat((position_info[i], props_duration),
                                      dim=-1).float()
            position_feats.append(
                self.position_transform(position_feat).permute(0, 2, 1))

        props_features = self.prop_fc(props_features)

        inputs = props_features.permute(0, 2, 1)
        outputs = self.backbone_net(inputs, query_features, position_feats)
        outputs = self.fpn(outputs)

        if self.is_second_stage:
            outputs = [_.detach() for _ in outputs]
        box_lists, loss_dict = self.fcos(outputs, gt_bbox.float())

        return box_lists, loss_dict
