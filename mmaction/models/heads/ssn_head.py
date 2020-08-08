import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS


def parse_stage_config(stage_cfg):
    if isinstance(stage_cfg, int):
        return (stage_cfg, ), stage_cfg
    elif isinstance(stage_cfg, (tuple, list)):
        return stage_cfg, sum(stage_cfg)
    else:
        raise ValueError(f'Incorrect STPP config {stage_cfg}')


class STPPTrain(nn.Module):

    def __init__(self,
                 with_standalong_classifier=False,
                 stpp_stage=(1, (1, 2), 1),
                 num_segments_list=(2, 5, 2)):
        super(STPPTrain, self).__init__()

        self.with_standalong_classifier = with_standalong_classifier

        starting_part, starting_multiplier = parse_stage_config(stpp_stage[0])
        course_part, course_multiplier = parse_stage_config(stpp_stage[1])
        ending_part, ending_multiplier = parse_stage_config(stpp_stage[2])

        self.num_multipliers = (
            starting_multiplier + course_multiplier + ending_multiplier)
        self.stpp_stages = (starting_part, course_part, ending_part)
        self.multiplier_list = (starting_multiplier, course_multiplier,
                                ending_multiplier)

        self.num_segments_list = num_segments_list

    def forward(self, x, scale_factors):
        x0 = self.num_segments_list[0]
        x1 = x0 + self.num_segments_list[1]
        num_segments = x1 + self.num_segments_list[2]

        feat_dim = x.size(1)
        x = x.view(-1, num_segments, feat_dim)
        num_samples = x.size(0)

        scale_factors = scale_factors.view(-1, 2)

        def extract_stpp_feature(stage_feat, stage_parts, num_multiplier,
                                 scale_factors):
            stage_stpp = []
            stage_len = stage_feat.size(1)
            for stage_part in stage_parts:
                ticks = torch.arange(0, stage_len + 1e-5,
                                     stage_len / stage_part)
                for i in range(stage_part):
                    part_feat = stage_feat[:,
                                           int(ticks[i]
                                               ):int(ticks[i + 1]), :].mean(
                                                   dim=1) / num_multiplier
                    if scale_factors is not None:
                        part_feat = (
                            part_feat * scale_factors.view(num_samples, 1))
                    stage_stpp.append(part_feat)
            return stage_stpp

        feature_parts = []
        feature_parts.extend(
            extract_stpp_feature(x[:, :x0, :], self.stpp_stages[0],
                                 self.multiplier_list[0], scale_factors[:, 0]))
        feature_parts.extend(
            extract_stpp_feature(x[:, x0:x1, :], self.stpp_stages[1],
                                 self.multiplier_list[1], None))
        feature_parts.extend(
            extract_stpp_feature(x[:, x1:, :], self.stpp_stages[2],
                                 self.multiplier_list[2], scale_factors[:, 1]))
        stpp_feat = torch.cat(feature_parts, dim=1)
        if not self.with_standalong_classifier:
            return stpp_feat, stpp_feat
        else:
            course_feat = x[:, x0:x1, :].mean(dim=1)
            return course_feat, stpp_feat


class STPPTest(nn.Module):

    def __init__(self,
                 num_classes,
                 with_standalong_classifier=False,
                 with_regression=True,
                 stpp_stage=(1, (1, 2), 1)):
        super(STPPTest, self).__init__()

        self.with_standalong_classifier = with_standalong_classifier

        self.activity_score_len = num_classes + 1
        self.complete_score_len = num_classes
        self.reg_score_len = num_classes * 2
        self.with_regression = with_regression

        starting_parts, starting_multiplier = parse_stage_config(stpp_stage[0])
        course_parts, course_multiplier = parse_stage_config(stpp_stage[1])
        ending_parts, ending_multiplier = parse_stage_config(stpp_stage[2])

        self.feat_multiplier = (
            starting_multiplier + course_multiplier + ending_multiplier)
        self.feat_dim = self.activity_score_len + \
            self.feat_multiplier * (self.complete_score_len +
                                    self.reg_score_len)
        self.stpp_stage = (starting_parts, course_parts, ending_parts)

        if self.with_standalong_classifier:
            self.activity_slice = slice(0, self.activity_score_len)
        else:
            self.activity_slice = slice(
                0, self.activity_score_len * self.feat_multiplier)

        self.complete_slice = slice(
            self.activity_slice.stop, self.activity_slice.stop +
            self.complete_score_len * self.feat_multiplier)
        self.reg_slice = slice(
            self.complete_slice.stop, self.complete_slice.stop +
            self.reg_score_len * self.feat_multiplier)

    def forward(self, x, proposal_ticks, scale_factors):
        assert x.size(1) == self.feat_dim
        num_ticks = proposal_ticks.size(0)

        out_activity_scores = torch.zeros((num_ticks, self.activity_score_len),
                                          dtype=x.dtype)
        raw_activity_scores = x[:, self.activity_slice]

        out_complete_scores = torch.zeros((num_ticks, self.complete_score_len),
                                          dtype=x.dtype)
        raw_complete_scores = x[:, self.complete_slice]

        if self.with_regression:
            out_reg_scores = torch.zeros((num_ticks, self.reg_score_len),
                                         dtype=x.dtype)
            raw_reg_scores = x[:, self.reg_slice]
        else:
            out_reg_scores = None
            raw_reg_scores = None

        def pyramids_pooling(out_scores, index, raw_scores, ticks,
                             scale_factors, score_len, stpp_stage):
            offset = 0
            for stage_idx, stage_cfg in enumerate(stpp_stage):
                if stage_idx == 0:
                    scale_factor = scale_factors[0]
                elif stage_idx == len(stpp_stage) - 1:
                    scale_factor = scale_factors[1]
                else:
                    scale_factor = 1.0

                num_stages = sum(stage_cfg)
                tick_left = float(ticks[stage_idx])
                tick_right = float(
                    max(ticks[stage_idx] + 1, ticks[stage_idx + 1]))

                if tick_right <= 0 or tick_left >= raw_scores.size(0):
                    offset += num_stages
                    continue
                for n_part in stage_cfg:
                    part_ticks = torch.arange(
                        tick_left,
                        tick_right + 1e-5, (tick_right - tick_left) / n_part,
                        dtype=torch.float)

                    for i in range(n_part):
                        part_tick_left = int(part_ticks[i])
                        part_tick_right = int(part_ticks[i + 1])
                        if part_tick_right - part_tick_left >= 1:
                            raw_score = raw_scores[
                                part_tick_left:part_tick_right,
                                offset * score_len:(offset + 1) * score_len]
                            raw_scale_score = raw_score.mean(
                                dim=0) * scale_factor
                            out_scores[
                                index, :] += raw_scale_score.detach().cpu()
                        offset += 1

        for i in range(num_ticks):
            ticks = proposal_ticks[i]
            if self.with_standalong_classifier:
                out_activity_scores[i, :] = raw_activity_scores[
                    ticks[1]:max(ticks[1] + 1, ticks[2]), :].mean(dim=0)
            else:
                pyramids_pooling(out_activity_scores, i, raw_activity_scores,
                                 ticks, scale_factors[i],
                                 self.activity_score_len, self.stpp_stage)

            pyramids_pooling(out_complete_scores, i, raw_complete_scores,
                             ticks, scale_factors[i], self.complete_score_len,
                             self.stpp_stage)

            if self.with_regression:
                pyramids_pooling(out_reg_scores, i, raw_reg_scores, ticks,
                                 scale_factors[i], self.reg_score_len,
                                 self.stpp_stage)

        return out_activity_scores, out_complete_scores, out_reg_scores


@HEADS.register_module
class SSNHead(nn.Module):
    """SSN's classification head."""

    def __init__(self,
                 dropout_ratio=0.8,
                 in_channels_activity=3072,
                 in_channels_complete=3072,
                 num_classes=20,
                 consensus=dict(
                     type='STPPTrain',
                     standalong_classifier=True,
                     stpp_cfg=(1, 1, 1),
                     num_seg=(2, 5, 2)),
                 with_bg=False,
                 with_reg=True,
                 init_std=0.001):

        super(SSNHead, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.in_channels_activity = in_channels_activity
        self.in_channels_complete = in_channels_complete
        if with_bg:
            self.num_classes = num_classes - 1
        else:
            self.num_classes = num_classes

        self.with_bg = with_bg
        self.with_reg = with_reg
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')
        if consensus_type == 'STPPTrain':
            self.consensus = STPPTrain(**consensus_)
        elif consensus_type == 'STPPTest':
            consensus_['num_classes'] = self.num_classes
            self.consensus = STPPTest(**consensus_)

        self.activity_fc = nn.Linear(in_channels_activity, num_classes + 1)
        self.completeness_fc = nn.Linear(in_channels_complete, num_classes)
        if self.with_reg:
            self.regressor_fc = nn.Linear(in_channels_complete,
                                          num_classes * 2)

    def init_weights(self):
        normal_init(self.activity_fc, std=self.init_std)
        normal_init(self.completeness_fc, std=self.init_std)
        if self.with_reg:
            normal_init(self.regressor_fc, std=self.init_std)

    def prepare_test_fc(self, stpp_feat_multiplier):
        # Reorganize the shape of fc to improve testing efficiency.
        in_features = self.activity_fc.in_features
        out_features = (
            self.activity_fc.out_features +
            self.completeness_fc.out_features * stpp_feat_multiplier)
        if self.with_reg:
            out_features += (
                self.regressor_fc.out_features * stpp_feat_multiplier)
        self.test_fc = nn.Linear(in_features, out_features)

        # Fetch weight and bias of the reorganized fc.
        complete_weight = self.completeness_fc.weight.data.view(
            self.completeness_fc.out_features, stpp_feat_multiplier,
            self.activity_fc.in_features).transpose(0, 1).contiguous().view(
                -1, self.activity_fc.in_features)
        complete_bias = self.completeness_fc.bias.data.view(1, -1).expand(
            stpp_feat_multiplier, self.completeness_fc.out_features
        ).contiguous().view(-1) / stpp_feat_multiplier

        weight = torch.cat((self.activity_fc.weight.data, complete_weight))
        bias = torch.cat((self.activity_fc.bias.data, complete_bias))

        if self.with_reg:
            reg_weight = self.regressor_fc.weight.data.view(
                self.regressor_fc.out_features,
                stpp_feat_multiplier, self.activity_fc.in_features).transpose(
                    0, 1).contiguous().view(-1, self.activity_fc.in_features)
            reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                stpp_feat_multiplier, self.regressor_fc.out_features
            ).contiguous().view(-1) / stpp_feat_multiplier
            weight = torch.cat((weight, reg_weight))
            bias = torch.cat((bias, reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias
        return True

    def forward(self, x, test_mode=False):
        if not test_mode:
            x, proposal_scale_factor = x
            activity_feat, completeness_feat = self.consensus(
                x, proposal_scale_factor)

            if self.dropout is not None:
                activity_feat = self.dropout(activity_feat)
                completeness_feat = self.dropout(completeness_feat)

            activity_scores = self.activity_fc(activity_feat)
            complete_scores = self.completeness_fc(completeness_feat)
            if self.with_reg:
                bbox_pred = self.regressor_fc(completeness_feat)
            else:
                bbox_pred = None
            bbox_pred = bbox_pred.view(-1, self.completeness_fc.out_features,
                                       2)
            return activity_scores, complete_scores, bbox_pred
        else:
            x, proposal_tick_list, scale_factor_list = x
            test_scores = self.test_fc(x)
            (activity_scores, completeness_scores,
             bbox_preds) = self.consensus(test_scores, proposal_tick_list,
                                          scale_factor_list)

            return (test_scores, activity_scores, completeness_scores,
                    bbox_preds)
