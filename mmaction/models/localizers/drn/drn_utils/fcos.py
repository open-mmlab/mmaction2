# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class FCOSHead(torch.nn.Module):

    def __init__(self, in_channels: int, fcos_num_class: int,
                 fcos_conv_layers: int, fcos_prior_prob: float,
                 is_second_stage: bool) -> None:
        super(FCOSHead, self).__init__()
        num_classes = fcos_num_class - 1

        cls_tower = []
        bbox_tower = []
        for i in range(fcos_conv_layers):
            cls_tower.append(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_tower.append(nn.BatchNorm1d(in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_tower.append(nn.BatchNorm1d(in_channels))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_logits = nn.Conv1d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1)

        self.bbox_pred = nn.Conv1d(
            in_channels, 2, kernel_size=3, stride=1, padding=1)

        self.mix_fc = nn.Sequential(
            nn.Conv1d(2 * in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(in_channels), nn.ReLU())

        self.iou_scores = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, 1, kernel_size=1, stride=1),
        )

        # initialization
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=0.01)
                torch.nn.init.constant_(module.bias, 0)

        # initialize the bias for focal loss
        bias_value = -math.log((1 - fcos_prior_prob) / fcos_prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(3)])
        self.is_second_stage = is_second_stage

    def forward(self, x):
        logits = []
        bbox_reg = []
        iou_scores = []
        for idx, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))

            bbox_reg_ = torch.exp(self.scales[idx](self.bbox_pred(box_tower)))
            if self.is_second_stage:
                bbox_reg_ = bbox_reg_.detach()
            bbox_reg.append(bbox_reg_)

            mix_feature = torch.cat([cls_tower, box_tower], dim=1)
            if self.is_second_stage:
                mix_feature = mix_feature.detach()
            mix_feature = self.mix_fc(mix_feature)
            iou_scores.append(self.iou_scores(mix_feature))
        return logits, bbox_reg, iou_scores


class FCOSModule(torch.nn.Module):

    def __init__(self, in_channels: int, fcos_num_class: int,
                 fcos_conv_layers: int, fcos_prior_prob: float,
                 fcos_inference_thr: float, fcos_pre_nms_top_n: int,
                 fcos_nms_thr: float, test_detections_per_img: int,
                 fpn_stride: int, focal_alpha: float, focal_gamma: float,
                 is_first_stage: bool, is_second_stage: bool) -> None:
        super(FCOSModule, self).__init__()

        head = FCOSHead(
            in_channels=in_channels,
            fcos_num_class=fcos_num_class,
            fcos_conv_layers=fcos_conv_layers,
            fcos_prior_prob=fcos_prior_prob,
            is_second_stage=is_second_stage)

        self.is_first_stage = is_first_stage
        self.is_second_stage = is_second_stage
        box_selector_test = make_fcos_postprocessor(fcos_num_class,
                                                    fcos_inference_thr,
                                                    fcos_pre_nms_top_n,
                                                    fcos_nms_thr,
                                                    test_detections_per_img,
                                                    is_first_stage)
        loss_evaluator = make_fcos_loss_evaluator(focal_alpha, focal_gamma)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = fpn_stride

    def forward(self, features, targets=None):
        box_cls, box_regression, iou_scores = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(locations, box_cls, box_regression,
                                       targets, iou_scores)
        else:
            return self._forward_test(locations, box_cls, box_regression,
                                      targets, iou_scores)

    def _forward_train(self, locations, box_cls, box_regression, targets,
                       iou_scores):
        loss_box_cls, loss_box_reg, loss_iou = self.loss_evaluator(
            locations, box_cls, box_regression, targets, iou_scores,
            self.is_first_stage)

        if self.is_second_stage:
            loss_box_cls = loss_box_cls.detach()
            loss_box_reg = loss_box_reg.detach()
        if self.is_first_stage:
            loss_iou = loss_iou.detach()

        losses = {
            'loss_cls': loss_box_cls,
            'loss_reg': loss_box_reg,
            'loss_iou': loss_iou
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, targets,
                      iou_scores):
        boxes = self.box_selector_test(locations, box_cls, box_regression,
                                       iou_scores)
        losses = None
        return boxes, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            t = feature.size(-1)
            locations_per_level = self.compute_locations_per_level(
                t, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, t, stride, device):
        shifts_t = torch.arange(
            0, t * stride, step=stride, dtype=torch.float32, device=device)
        shifts_t = shifts_t.reshape(-1)
        locations = shifts_t + stride / 2
        return locations
