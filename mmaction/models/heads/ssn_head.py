import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS


def parse_stage_config(stage_cfg):
    """Parse config of STPP for three stages.

    Args:
        stage_cfg (int | tuple[int]):
            Config of structured temporal pyramid pooling.

    Returns:
        tuple[tuple[int], int]:
            Config of structured temporal pyramid pooling and
            total number of parts(number of multipliers).
    """
    if isinstance(stage_cfg, int):
        return (stage_cfg, ), stage_cfg
    if isinstance(stage_cfg, tuple):
        return stage_cfg, sum(stage_cfg)
    raise ValueError(f'Incorrect STPP config {stage_cfg}')


class STPPTrain(nn.Module):
    """Structured temporal pyramid pooling for SSN at training.

    Args:
        stpp_stage (tuple): Config of structured temporal pyramid pooling.
            Default: (1, (1, 2), 1).
        num_segments_list (tuple): Number of segments to be sampled
            in three stages. Default: (2, 5, 2).
    """

    def __init__(self, stpp_stage=(1, (1, 2), 1), num_segments_list=(2, 5, 2)):
        super().__init__()

        starting_part, starting_multiplier = parse_stage_config(stpp_stage[0])
        course_part, course_multiplier = parse_stage_config(stpp_stage[1])
        ending_part, ending_multiplier = parse_stage_config(stpp_stage[2])

        self.num_multipliers = (
            starting_multiplier + course_multiplier + ending_multiplier)
        self.stpp_stages = (starting_part, course_part, ending_part)
        self.multiplier_list = (starting_multiplier, course_multiplier,
                                ending_multiplier)

        self.num_segments_list = num_segments_list

    @staticmethod
    def _extract_stage_feature(stage_feat, stage_parts, num_multipliers,
                               scale_factors, num_samples):
        """Extract stage feature based on structured temporal pyramid pooling.

        Args:
            stage_feat (torch.Tensor): Stage features to be STPP.
            stage_parts (tuple): Config of STPP.
            num_multipliers (int): Total number of parts in the stage.
            scale_factors (list): Ratios of the effective sampling lengths
                to augmented lengths.
            num_samples (int): Number of samples.

        Returns:
            torch.Tensor: Features of the stage.
        """
        stage_stpp_feat = []
        stage_len = stage_feat.size(1)
        for stage_part in stage_parts:
            ticks = torch.arange(0, stage_len + 1e-5,
                                 stage_len / stage_part).int()
            for i in range(stage_part):
                part_feat = stage_feat[:, ticks[i]:ticks[i + 1], :].mean(
                    dim=1) / num_multipliers
                if scale_factors is not None:
                    part_feat = (
                        part_feat * scale_factors.view(num_samples, 1))
                stage_stpp_feat.append(part_feat)
        return stage_stpp_feat

    def forward(self, x, scale_factors):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            scale_factors (list): Ratios of the effective sampling lengths
                to augmented lengths.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                Features for predicting activity scores and
                completeness scores.
        """
        x0 = self.num_segments_list[0]
        x1 = x0 + self.num_segments_list[1]
        num_segments = x1 + self.num_segments_list[2]

        feat_dim = x.size(1)
        x = x.view(-1, num_segments, feat_dim)
        num_samples = x.size(0)

        scale_factors = scale_factors.view(-1, 2)

        stage_stpp_feats = []
        stage_stpp_feats.extend(
            self._extract_stage_feature(x[:, :x0, :], self.stpp_stages[0],
                                        self.multiplier_list[0],
                                        scale_factors[:, 0], num_samples))
        stage_stpp_feats.extend(
            self._extract_stage_feature(x[:, x0:x1, :], self.stpp_stages[1],
                                        self.multiplier_list[1], None,
                                        num_samples))
        stage_stpp_feats.extend(
            self._extract_stage_feature(x[:, x1:, :], self.stpp_stages[2],
                                        self.multiplier_list[2],
                                        scale_factors[:, 1], num_samples))
        stpp_feat = torch.cat(stage_stpp_feats, dim=1)

        course_feat = x[:, x0:x1, :].mean(dim=1)
        return course_feat, stpp_feat


class STPPTest(nn.Module):
    """Structured temporal pyramid pooling for SSN at testing.

    Args:
        num_classes (int): Number of classes to be classified.
        use_regression (bool): Whether to perform regression or not.
            Default: True.
        stpp_stage (tuple): Config of structured temporal pyramid pooling.
            Default: (1, (1, 2), 1).
    """

    def __init__(self,
                 num_classes,
                 use_regression=True,
                 stpp_stage=(1, (1, 2), 1)):
        super().__init__()

        self.activity_score_len = num_classes + 1
        self.complete_score_len = num_classes
        self.reg_score_len = num_classes * 2
        self.use_regression = use_regression

        starting_parts, starting_multiplier = parse_stage_config(stpp_stage[0])
        course_parts, course_multiplier = parse_stage_config(stpp_stage[1])
        ending_parts, ending_multiplier = parse_stage_config(stpp_stage[2])

        self.num_multipliers = (
            starting_multiplier + course_multiplier + ending_multiplier)
        if self.use_regression:
            self.feat_dim = (
                self.activity_score_len + self.num_multipliers *
                (self.complete_score_len + self.reg_score_len))
        else:
            self.feat_dim = (
                self.activity_score_len +
                self.num_multipliers * self.complete_score_len)
        self.stpp_stage = (starting_parts, course_parts, ending_parts)

        self.activity_slice = slice(0, self.activity_score_len)
        self.complete_slice = slice(
            self.activity_slice.stop, self.activity_slice.stop +
            self.complete_score_len * self.num_multipliers)
        self.reg_slice = slice(
            self.complete_slice.stop, self.complete_slice.stop +
            self.reg_score_len * self.num_multipliers)

    @staticmethod
    def _pyramids_pooling(out_scores, index, raw_scores, ticks, scale_factors,
                          score_len, stpp_stage):
        """Perform pyramids pooling.

        Args:
            out_scores (torch.Tensor): Scores to be returned.
            index (int): Index of output scores.
            raw_scores (torch.Tensor): Raw scores before STPP.
            ticks (list): Ticks of raw scores.
            scale_factors (list): Ratios of the effective sampling lengths
                to augmented lengths.
            score_len (int): Length of the score.
            stpp_stage (tuple): Config of STPP.
        """
        offset = 0
        for stage_idx, stage_cfg in enumerate(stpp_stage):
            if stage_idx == 0:
                scale_factor = scale_factors[0]
            elif stage_idx == len(stpp_stage) - 1:
                scale_factor = scale_factors[1]
            else:
                scale_factor = 1.0

            sum_parts = sum(stage_cfg)
            tick_left = ticks[stage_idx]
            tick_right = float(max(ticks[stage_idx] + 1, ticks[stage_idx + 1]))

            if tick_right <= 0 or tick_left >= raw_scores.size(0):
                offset += sum_parts
                continue
            for num_parts in stage_cfg:
                part_ticks = torch.arange(tick_left, tick_right + 1e-5,
                                          (tick_right - tick_left) /
                                          num_parts).int()

                for i in range(num_parts):
                    part_tick_left = part_ticks[i]
                    part_tick_right = part_ticks[i + 1]
                    if part_tick_right - part_tick_left >= 1:
                        raw_score = raw_scores[part_tick_left:part_tick_right,
                                               offset *
                                               score_len:(offset + 1) *
                                               score_len]
                        raw_scale_score = raw_score.mean(dim=0) * scale_factor
                        out_scores[index, :] += raw_scale_score.detach().cpu()
                    offset += 1

        return out_scores

    def forward(self, x, proposal_ticks, scale_factors):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            proposal_ticks (list): Ticks of proposals to be STPP.
            scale_factors (list): Ratios of the effective sampling lengths
                to augmented lengths.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                out_activity_scores (torch.Tensor): Activity scores
                out_complete_scores (torch.Tensor): Completeness scores.
                out_reg_scores (torch.Tensor): Regression scores.
        """
        assert x.size(1) == self.feat_dim
        num_ticks = proposal_ticks.size(0)

        out_activity_scores = torch.zeros((num_ticks, self.activity_score_len),
                                          dtype=x.dtype)
        raw_activity_scores = x[:, self.activity_slice]

        out_complete_scores = torch.zeros((num_ticks, self.complete_score_len),
                                          dtype=x.dtype)
        raw_complete_scores = x[:, self.complete_slice]

        if self.use_regression:
            out_reg_scores = torch.zeros((num_ticks, self.reg_score_len),
                                         dtype=x.dtype)
            raw_reg_scores = x[:, self.reg_slice]
        else:
            out_reg_scores = None
            raw_reg_scores = None

        for i in range(num_ticks):
            ticks = proposal_ticks[i]

            out_activity_scores[i, :] = raw_activity_scores[
                ticks[1]:max(ticks[1] + 1, ticks[2]), :].mean(dim=0)

            out_complete_scores = self._pyramids_pooling(
                out_complete_scores, i, raw_complete_scores, ticks,
                scale_factors[i], self.complete_score_len, self.stpp_stage)

            if self.use_regression:
                out_reg_scores = self._pyramids_pooling(
                    out_reg_scores, i, raw_reg_scores, ticks, scale_factors[i],
                    self.reg_score_len, self.stpp_stage)

        return out_activity_scores, out_complete_scores, out_reg_scores


@HEADS.register_module()
class SSNHead(nn.Module):
    """The classification head for SSN.

    Args:
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        in_channels (int): Number of channels for input data. Default: 1024.
        num_classes (int): Number of classes to be classified. Default: 20.
        consensus (dict): Config of segmental consensus.
        use_regression (bool): Whether to perform regression or not.
            Default: True.
        init_std (float): Std value for Initiation. Default: 0.001.
    """

    def __init__(self,
                 dropout_ratio=0.8,
                 in_channels=1024,
                 num_classes=20,
                 consensus=dict(
                     type='STPPTrain',
                     standalong_classifier=True,
                     stpp_cfg=(1, 1, 1),
                     num_seg=(2, 5, 2)),
                 use_regression=True,
                 init_std=0.001):

        super().__init__()

        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.use_regression = use_regression
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # Based on this copy, the model will utilize different
        # structured temporal pyramid pooling at training and testing.
        # Warning: this copy cannot be removed.
        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')
        if consensus_type == 'STPPTrain':
            self.consensus = STPPTrain(**consensus_)
        elif consensus_type == 'STPPTest':
            consensus_['num_classes'] = self.num_classes
            self.consensus = STPPTest(**consensus_)

        self.in_channels_activity = in_channels
        self.in_channels_complete = (
            self.consensus.num_multipliers * in_channels)
        self.activity_fc = nn.Linear(in_channels, num_classes + 1)
        self.completeness_fc = nn.Linear(self.in_channels_complete,
                                         num_classes)
        if self.use_regression:
            self.regressor_fc = nn.Linear(self.in_channels_complete,
                                          num_classes * 2)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.activity_fc, std=self.init_std)
        normal_init(self.completeness_fc, std=self.init_std)
        if self.use_regression:
            normal_init(self.regressor_fc, std=self.init_std)

    def prepare_test_fc(self, stpp_feat_multiplier):
        """Reorganize the shape of fully connected layer at testing, in order
        to improve testing efficiency.

        Args:
            stpp_feat_multiplier (int): Total number of parts.

        Returns:
            bool: Whether the shape transformation is ready for testing.
        """

        in_features = self.activity_fc.in_features
        out_features = (
            self.activity_fc.out_features +
            self.completeness_fc.out_features * stpp_feat_multiplier)
        if self.use_regression:
            out_features += (
                self.regressor_fc.out_features * stpp_feat_multiplier)
        self.test_fc = nn.Linear(in_features, out_features)

        # Fetch weight and bias of the reorganized fc.
        complete_weight = self.completeness_fc.weight.data.view(
            self.completeness_fc.out_features, stpp_feat_multiplier,
            in_features).transpose(0, 1).contiguous().view(-1, in_features)
        complete_bias = self.completeness_fc.bias.data.view(1, -1).expand(
            stpp_feat_multiplier, self.completeness_fc.out_features
        ).contiguous().view(-1) / stpp_feat_multiplier

        weight = torch.cat((self.activity_fc.weight.data, complete_weight))
        bias = torch.cat((self.activity_fc.bias.data, complete_bias))

        if self.use_regression:
            reg_weight = self.regressor_fc.weight.data.view(
                self.regressor_fc.out_features, stpp_feat_multiplier,
                in_features).transpose(0,
                                       1).contiguous().view(-1, in_features)
            reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                stpp_feat_multiplier, self.regressor_fc.out_features
            ).contiguous().view(-1) / stpp_feat_multiplier
            weight = torch.cat((weight, reg_weight))
            bias = torch.cat((bias, reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias
        return True

    def forward(self, x, test_mode=False):
        """Defines the computation performed at every call."""
        if not test_mode:
            x, proposal_scale_factor = x
            activity_feat, completeness_feat = self.consensus(
                x, proposal_scale_factor)

            if self.dropout is not None:
                activity_feat = self.dropout(activity_feat)
                completeness_feat = self.dropout(completeness_feat)

            activity_scores = self.activity_fc(activity_feat)
            complete_scores = self.completeness_fc(completeness_feat)
            if self.use_regression:
                bbox_preds = self.regressor_fc(completeness_feat)
                bbox_preds = bbox_preds.view(-1,
                                             self.completeness_fc.out_features,
                                             2)
            else:
                bbox_preds = None
            return activity_scores, complete_scores, bbox_preds

        x, proposal_tick_list, scale_factor_list = x
        test_scores = self.test_fc(x)
        (activity_scores, completeness_scores,
         bbox_preds) = self.consensus(test_scores, proposal_tick_list,
                                      scale_factor_list)

        return (test_scores, activity_scores, completeness_scores, bbox_preds)
