import os.path as osp

import numpy as np

from .proposal_utils import temporal_iop, temporal_iou


def generate_candidate_proposals(video_list,
                                 video_infos,
                                 tem_results_dir,
                                 temporal_scale,
                                 peak_threshold,
                                 tem_results_ext='.csv',
                                 result_dict=None):
    """Generate Candidate Proposals with given temporal evalutation results.
    Each proposal file will contain:
    'tmin,tmax,tmin_score,tmax_score,score,match_iou,match_ioa'.

    Args:
        video_list (list[int]): List of video indexs to generate proposals.
        video_infos (list[dict]): List of video_info dict that contains
            'video_name', 'duration_frame', 'duration_second',
            'feature_frame', and 'annotations'.
        tem_results_dir (str): Directory to load temporal evaluation
            results.
        temporal_scale (int): The number (scale) on temporal axis.
        peak_threshold (float): The threshold for proposal generation.
        tem_results_ext (str): File extension for temporal evaluation
            model output. Default: '.csv'.
        result_dict (dict): The dict to save the results. Default: None.

    Returns:
        dict: A dict contains video_name as keys and proposal list as value.
        If result_dict is not None, save the results to it.
    """
    if tem_results_ext not in ('.csv'):
        raise NotImplementedError

    tscale = temporal_scale
    tgap = 1. / tscale
    proposal_dict = {}
    for video_index in video_list:
        video_name = video_infos[video_index]['video_name']
        tem_path = osp.join(tem_results_dir, video_name + tem_results_ext)
        tem_results = np.loadtxt(
            tem_path, dtype=np.float32, delimiter=',', skiprows=1)
        start_scores = tem_results[:, 1]
        end_scores = tem_results[:, 2]

        max_start = max(start_scores)
        max_end = max(end_scores)

        start_bins = np.zeros(len(start_scores))
        start_bins[[0, -1]] = 1
        end_bins = np.zeros(len(end_scores))
        end_bins[[0, -1]] = 1
        for idx in range(1, tscale - 1):
            if start_scores[idx] > start_scores[
                    idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                start_bins[idx] = 1
            elif start_scores[idx] > (peak_threshold * max_start):
                start_bins[idx] = 1
            if end_scores[idx] > end_scores[
                    idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                end_bins[idx] = 1
            elif end_scores[idx] > (peak_threshold * max_end):
                end_bins[idx] = 1

        tmin_list = []
        tmin_score_list = []
        tmax_list = []
        tmax_score_list = []
        for idx in range(tscale):
            if start_bins[idx] == 1:
                tmin_list.append(tgap / 2 + tgap * idx)
                tmin_score_list.append(start_scores[idx])
            if end_bins[idx] == 1:
                tmax_list.append(tgap / 2 + tgap * idx)
                tmax_score_list.append(end_scores[idx])

        new_props = []
        for tmax, tmax_score in zip(tmax_list, tmax_score_list):
            for tmin, tmin_score in zip(tmin_list, tmin_score_list):
                if tmin >= tmax:
                    break
                new_props.append([tmin, tmax, tmin_score, tmax_score])

        new_props = np.stack(new_props)

        score = (new_props[:, 2] * new_props[:, 3]).reshape(-1, 1)
        new_props = np.concatenate((new_props, score), axis=1)

        new_props = new_props[new_props[:, -1].argsort()[::-1]]
        video_info = video_infos[video_index]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second

        gt_tmins = []
        gt_tmaxs = []
        for annotations in video_info['annotations']:
            gt_tmins.append(annotations['segment'][0] / corrected_second)
            gt_tmaxs.append(annotations['segment'][1] / corrected_second)

        new_iou_list = []
        new_ioa_list = []
        for new_prop in new_props:
            new_iou = max(
                temporal_iou(new_prop[0], new_prop[1], gt_tmins, gt_tmaxs))
            new_ioa = max(
                temporal_iop(new_prop[0], new_prop[1], gt_tmins, gt_tmaxs))
            new_iou_list.append(new_iou)
            new_ioa_list.append(new_ioa)

        new_iou_list = np.array(new_iou_list).reshape(-1, 1)
        new_ioa_list = np.array(new_ioa_list).reshape(-1, 1)
        new_props = np.concatenate((new_props, new_iou_list), axis=1)
        new_props = np.concatenate((new_props, new_ioa_list), axis=1)
        proposal_dict[video_name] = new_props
        if result_dict is not None:
            result_dict[video_name] = new_props
    return proposal_dict


def generate_bsp_feature(video_list,
                         video_infos,
                         tem_results_dir,
                         pgm_proposals_dir,
                         top_k=1000,
                         bsp_boundary_ratio=0.2,
                         num_sample_start=8,
                         num_sample_end=8,
                         num_sample_action=16,
                         num_sample_interp=3,
                         tem_results_ext='.csv',
                         pgm_proposal_ext='.csv',
                         result_dict=None):
    """Generate Boundary-Sensitive Proposal Feature with given proposals.

    Args:
        video_list (list[int]): List of video indexs to generate bsp_feature.
        video_infos (list[dict]): List of video_info dict that contains
            'video_name'.
        tem_results_dir (str): Directory to load temporal evaluation
            results.
        pgm_proposals_dir (str): Directory to load proposals.
        top_k (int): Number of proposals to be considered. Default: 1000
        bsp_boundary_ratio (float): Ratio for proposal boundary
            (start/end). Default: 0.2.
        num_sample_start (int): Num of samples for actionness in
            start region. Default: 8.
        num_sample_end (int): Num of samples for actionness in end region.
            Default: 8.
        num_sample_action (int): Num of samples for actionness in center
            region. Default: 16.
        num_sample_interp (int): Num of samples for interpolation for
            each sample point. Default: 3.
        tem_results_ext (str): File extension for temporal evaluation
            model output. Default: '.csv'.
        pgm_proposal_ext (str): File extension for proposals. Default: '.csv'.
        result_dict (dict): The dict to save the results. Default: None.

    Returns:
        bsp_feature_dict (dict): A dict contains video_name as keys and
        bsp_feature as value. If result_dict is not None, save the
        results to it.
    """
    if tem_results_ext not in ('.csv') or pgm_proposal_ext not in ('.csv'):
        raise NotImplementedError

    bsp_feature_dict = {}
    for video_index in video_list:
        video_name = video_infos[video_index]['video_name']

        # Load temporal evaluation results
        tem_path = osp.join(tem_results_dir, video_name + tem_results_ext)
        tem_results = np.loadtxt(
            tem_path, dtype=np.float32, delimiter=',', skiprows=1)
        score_action = tem_results[:, 0]
        seg_tmins = tem_results[:, 3]
        seg_tmaxs = tem_results[:, 4]
        video_scale = len(tem_results)
        video_gap = seg_tmaxs[0] - seg_tmins[0]
        video_extend = int(video_scale / 4 + 10)

        # Load proposals results
        proposal_path = osp.join(pgm_proposals_dir,
                                 video_name + pgm_proposal_ext)
        pgm_proposals = np.loadtxt(
            proposal_path, dtype=np.float32, delimiter=',', skiprows=1)
        pgm_proposals = pgm_proposals[:top_k]

        # Generate temporal sample points
        boundary_zeros = np.zeros([video_extend])
        score_action = np.concatenate(
            (boundary_zeros, score_action, boundary_zeros))
        begin_tp = []
        middle_tp = []
        end_tp = []
        for i in range(video_extend):
            begin_tp.append(-video_gap / 2 -
                            (video_extend - 1 - i) * video_gap)
            end_tp.append(video_gap / 2 + seg_tmaxs[-1] + i * video_gap)
        for i in range(video_scale):
            middle_tp.append(video_gap / 2 + i * video_gap)
        t_points = begin_tp + middle_tp + end_tp

        bsp_feature = []
        for pgm_proposal in pgm_proposals:
            tmin = pgm_proposal[0]
            tmax = pgm_proposal[1]

            tlen = tmax - tmin
            # Temporal range for start
            tmin_0 = tmin - tlen * bsp_boundary_ratio
            tmin_1 = tmin + tlen * bsp_boundary_ratio
            # Temporal range for end
            tmax_0 = tmax - tlen * bsp_boundary_ratio
            tmax_1 = tmax + tlen * bsp_boundary_ratio

            # Generate features at start boundary
            tlen_start = (tmin_1 - tmin_0) / (num_sample_start - 1)
            tlen_start_sample = tlen_start / num_sample_interp
            t_new = [
                tmin_0 - tlen_start / 2 + tlen_start_sample * i
                for i in range(num_sample_start * num_sample_interp + 1)
            ]
            y_new_start_action = np.interp(t_new, t_points, score_action)
            y_new_start = [
                np.mean(y_new_start_action[i * num_sample_interp:(i + 1) *
                                           num_sample_interp + 1])
                for i in range(num_sample_start)
            ]
            # Generate features at end boundary
            tlen_end = (tmax_1 - tmax_0) / (num_sample_end - 1)
            tlen_end_sample = tlen_end / num_sample_interp
            t_new = [
                tmax_0 - tlen_end / 2 + tlen_end_sample * i
                for i in range(num_sample_end * num_sample_interp + 1)
            ]
            y_new_end_action = np.interp(t_new, t_points, score_action)
            y_new_end = [
                np.mean(y_new_end_action[i * num_sample_interp:(i + 1) *
                                         num_sample_interp + 1])
                for i in range(num_sample_end)
            ]
            # Generate features for action
            tlen_action = (tmax - tmin) / (num_sample_action - 1)
            tlen_action_sample = tlen_action / num_sample_interp
            t_new = [
                tmin - tlen_action / 2 + tlen_action_sample * i
                for i in range(num_sample_action * num_sample_interp + 1)
            ]
            y_new_action = np.interp(t_new, t_points, score_action)
            y_new_action = [
                np.mean(y_new_action[i * num_sample_interp:(i + 1) *
                                     num_sample_interp + 1])
                for i in range(num_sample_action)
            ]
            feature = np.concatenate([y_new_action, y_new_start, y_new_end])
            bsp_feature.append(feature)
        bsp_feature = np.array(bsp_feature)
        bsp_feature_dict[video_name] = bsp_feature
        if result_dict is not None:
            result_dict[video_name] = bsp_feature
    return bsp_feature_dict
