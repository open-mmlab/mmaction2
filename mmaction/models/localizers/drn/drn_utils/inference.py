# Copyright (c) OpenMMLab. All rights reserved.
"""Copied from https://github.com/Alvin-Zeng/DRN/"""

import torch


class FCOSPostProcessor(torch.nn.Module):
    """Performs post-processing on the outputs of the RetinaNet boxes.

    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
                 fpn_post_nms_top_n, min_size, num_classes, is_first_stage):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.innerness_threshold = 0.15
        self.downsample_scale = 32
        self.is_first_stage = is_first_stage

    def forward_for_single_feature_map(self, locations, box_cls,
                                       box_regression, level, iou_scores):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, T = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.permute(0, 2, 1).contiguous().sigmoid()
        iou_scores = iou_scores.permute(0, 2, 1).contiguous().sigmoid()
        box_regression = box_regression.permute(0, 2, 1)

        # centerness = centerness.permute(0, 2, 1)
        # centerness = centerness.reshape(N, -1).sigmoid()
        # inner = inner.squeeze().sigmoid()

        candidate_inds = (box_cls > self.pre_nms_thresh)
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # box_cls = box_cls * centerness[:, :, None]
        # box_cls = box_cls + centerness[:, :, None]
        if not self.is_first_stage:
            box_cls = box_cls * iou_scores

        results = []
        for i in range(N):

            # per_centerness = centerness[i]

            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            # per_centerness = per_centerness[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

                # per_centerness = per_centerness[top_k_indices]

            detections = torch.stack([
                per_locations - per_box_regression[:, 0],
                per_locations + per_box_regression[:, 1],
            ],
                                     dim=1) / self.downsample_scale

            detections[:, 0].clamp_(min=0, max=1)
            detections[:, 1].clamp_(min=0, max=1)

            # remove small boxes
            p_start, p_end = detections.unbind(dim=1)
            duration = p_end - p_start
            keep = (duration >= self.min_size).nonzero().squeeze(1)
            detections = detections[keep]

            temp_dict = {}
            temp_dict['detections'] = detections
            temp_dict['labels'] = per_class
            temp_dict['scores'] = torch.sqrt(per_box_cls)
            temp_dict['level'] = [level]
            # temp_dict['centerness'] = per_centerness
            temp_dict['locations'] = per_locations / 32

            results.append(temp_dict)

        return results

    def forward(self, locations, box_cls, box_regression, iou_scores):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for i, (l, o, b, iou_s) in enumerate(
                zip(locations, box_cls, box_regression, iou_scores)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(l, o, b, i, iou_s))

        boxlists = list(zip(*sampled_boxes))
        # boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            dicts = boxlists[i]
            per_vid_scores = []
            per_vid_detections = []
            per_vid_labels = []
            # add level number
            per_vid_level = []
            per_vid_locations = []
            # per_vid_centerness = []
            for per_scale_dict in dicts:
                if len(per_scale_dict['detections']) != 0:
                    per_vid_detections.append(per_scale_dict['detections'])
                if len(per_scale_dict['scores']) != 0:
                    per_vid_scores.append(per_scale_dict['scores'])
                if len(per_scale_dict['level']) != 0:
                    per_vid_level.append(per_scale_dict['level'] *
                                         len(per_scale_dict['detections']))

                if len(per_scale_dict['locations']) != 0:
                    per_vid_locations.append(per_scale_dict['locations'])

                # if len(per_scale_dict['centerness']) != 0:
                #     per_vid_centerness.append(per_scale_dict['centerness'])
            if len(per_vid_detections) == 0:
                per_vid_detections = torch.Tensor([0, 1]).unsqueeze(0)
                per_vid_scores = torch.Tensor([1])
                per_vid_level = [[-1]]
                per_vid_locations = torch.Tensor([0.5])
                # per_vid_centerness = torch.Tensor([0.5]).cuda()
            else:
                per_vid_detections = torch.cat(per_vid_detections, dim=0)
                per_vid_scores = torch.cat(per_vid_scores, dim=0)
                per_vid_level = per_vid_level
                per_vid_locations = torch.cat(per_vid_locations, dim=0)
                # per_vid_centerness = torch.cat(per_vid_centerness, dim=0)

            temp_dict = {}
            temp_dict['detections'] = per_vid_detections
            temp_dict['labels'] = per_vid_labels
            temp_dict['scores'] = per_vid_scores
            temp_dict['level'] = per_vid_level
            # temp_dict['centerness'] = per_vid_centerness
            temp_dict['locations'] = per_vid_locations
            results.append(temp_dict)

        return results


def make_fcos_postprocessor(fcos_num_class, fcos_inference_thr,
                            fcos_pre_nms_top_n, fcos_nms_thr,
                            test_detections_per_img, is_first_stage):
    box_selector = FCOSPostProcessor(
        pre_nms_thresh=fcos_inference_thr,
        pre_nms_top_n=fcos_pre_nms_top_n,
        nms_thresh=fcos_nms_thr,
        fpn_post_nms_top_n=test_detections_per_img,
        min_size=0,
        num_classes=fcos_num_class,
        is_first_stage=is_first_stage)

    return box_selector
