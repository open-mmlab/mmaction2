import fnmatch
import glob
import os
import os.path as osp
from itertools import groupby

import numpy as np

from ..core import compute_average_precision_detection, np_softmax
from . import temporal_iou


def load_localize_proposal_file(filename):
    """Load the proposal file and split it into many parts which contain one
    video's information separately.

    Args:
        filename(str): Path to the proposal file.

    Returns:
        list: List of all videos' information.
    """
    lines = list(open(filename))

    # Split the proposal file into many parts which contain one video's
    # information separately.
    groups = groupby(lines, lambda x: x.startswith('#'))

    video_infos = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(video_info):
        """Parse the video's information.

        Template information of a video in a standard file:
            # index
            video_id
            num_frames
            fps
            num_gts
            label, start_frame, end_frame
            label, start_frame, end_frame
            ...
            num_proposals
            label, best_iou, overlap_self, start_frame, end_frame
            label, best_iou, overlap_self, start_frame, end_frame
            ...

        Example of a standard annotation file:
        .. code-block:: txt
            # 0
            video_validation_0000202
            5666
            1
            3
            8 130 185
            8 832 1136
            8 1303 1381
            5
            8 0.0620 0.0620 790 5671
            8 0.1656 0.1656 790 2619
            8 0.0833 0.0833 3945 5671
            8 0.0960 0.0960 4173 5671
            8 0.0614 0.0614 3327 5671

        Args:
            video_info (list): Information of the video.

        Returns:
            tuple[str, int, list, list]:
                video_id (str): Name of the video.
                num_frames (int): Number of frames in the video.
                gt_boxes (list): List of the information of gt boxes.
                proposal_boxes (list): List of the information of
                    proposal boxes.
        """
        offset = 0
        video_id = video_info[offset]
        offset += 1

        num_frames = int(float(video_info[1]) * float(video_info[2]))
        num_gts = int(video_info[3])
        offset = 4

        gt_boxes = [x.split() for x in video_info[offset:offset + num_gts]]
        offset += num_gts
        num_proposals = int(video_info[offset])
        offset += 1
        proposal_boxes = [
            x.split() for x in video_info[offset:offset + num_proposals]
        ]

        return video_id, num_frames, gt_boxes, proposal_boxes

    return [parse_group(video_info) for video_info in video_infos]


def process_norm_proposal_file(norm_proposal_file, proposal_file, frame_dict):
    """Process the normalized proposal file and denormalize it.

    Args:
        norm_proposal_file (str): Name of normalized proposal file.
        proposal_file (str): Name of proposal file.
        frame_dict (dict): Information of frame folders.
    """
    norm_proposals = load_localize_proposal_file(norm_proposal_file)

    processed_proposal_list = []
    for idx, norm_proposal in enumerate(norm_proposals):
        video_id = norm_proposal[0]
        frame_info = frame_dict[video_id]
        num_frames = frame_info[1]
        frame_path = osp.basename(frame_info[0])

        gt = [[
            int(x[0]),
            int(float(x[1]) * num_frames),
            int(float(x[2]) * num_frames)
        ] for x in norm_proposal[2]]

        proposal = [[
            int(x[0]),
            float(x[1]),
            float(x[2]),
            int(float(x[3]) * num_frames),
            int(float(x[4]) * num_frames)
        ] for x in norm_proposal[3]]

        gt_dump = '\n'.join(['{} {} {}'.format(*x) for x in gt])
        gt_dump += '\n' if len(gt) else ''
        proposal_dump = '\n'.join(
            ['{} {:.04f} {:.04f} {} {}'.format(*x) for x in proposal])
        proposal_dump += '\n' if len(proposal) else ''

        processed_proposal_list.append(
            f'# {idx}\n{frame_path}\n{num_frames}\n1'
            f'\n{len(gt)}\n{gt_dump}{len(proposal)}\n{proposal_dump}')

    with open(proposal_file, 'w') as f:
        f.writelines(processed_proposal_list)


def parse_frame_folder(path,
                       rgb_prefix='img_',
                       flow_x_prefix='flow_x_',
                       flow_y_prefix='flow_y_',
                       level=1):
    """Parse directories holding extracted frames from standard benchmarks.

    Args:
        path (str): Path of frame root folder.
        rgb_prefix (str): Prefix of rgb frames. Default: 'img_'.
        flow_x_prefix (str): Prefix of flow x frames. Default: 'flow_x_'.
        flow_y_prefix (str): Prefix of flow y frames. Default: 'flow_y_'.
        level (int): Directory level of data.

    Returns:
        dict: Information of frame folders.
    """
    print(f'parse frames under folder {path}')
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory, prefix_list):
        frames = os.listdir(directory)
        num_list = [len(fnmatch.filter(frames, x + '*')) for x in prefix_list]
        return num_list

    # check RGB
    frame_dict = {}
    for i, frame_folder in enumerate(frame_folders):
        num_all = count_files(frame_folder,
                              (rgb_prefix, flow_x_prefix, flow_y_prefix))
        key = osp.basename(frame_folder)

        num_flow_x = num_all[1]
        num_flow_y = num_all[2]
        if num_flow_x != num_flow_y:
            raise ValueError('x and y direction have different number '
                             'of flow images. video: ' + frame_folder)
        if i % 200 == 0:
            print(f'{i} videos parsed')

        frame_dict[key] = (frame_folder, num_all[0], num_flow_x)

    print('Frame folder analysis done')
    return frame_dict


def results_to_detections(dataset,
                          results,
                          top_k=2000,
                          softmax_before_filter=True,
                          cls_top_k=2,
                          **kwargs):
    """Convert prediction results into detections.

    Args:
        dataset (:obj:`SSNDataset`): Instance of SSNDataset.
        results (list): Prediction results.
        top_k (int): Number of top results. Default: 2000.
        softmax_before_filter (bool): Whether to perform softmax operations
            before filtering results. Default: True.
        cls_top_k (int): Number of top results for each class. Default: 2.

    Returns:
        list: Detection results.
    """
    num_classes = results[0][1].shape[1] - 1
    detections = [dict()] * num_classes

    for idx in range(len(dataset)):
        video_id = dataset.video_infos[idx]['video_id']
        relative_proposals = results[idx][0]
        if len(relative_proposals[0].shape) == 3:
            relative_proposals = np.squeeze(relative_proposals, 0)

        action_scores = results[idx][1]
        complete_scores = results[idx][2]
        regression_scores = results[idx][3]
        if regression_scores is None:
            regression_scores = np.zeros(
                len(relative_proposals), num_classes, 2, dtype=np.float32)
        regression_scores = regression_scores.reshape((-1, num_classes, 2))

        if top_k <= 0:
            combined_scores = (
                np_softmax(action_scores[:, 1:], dim=1) *
                np.exp(complete_scores))
            for i in range(num_classes):
                center_scores = regression_scores[:, i, 0][:, None]
                duration_scores = regression_scores[:, i, 1][:, None]
                detections[i][video_id] = np.concatenate(
                    (relative_proposals, combined_scores[:, i][:, None],
                     center_scores, duration_scores),
                    axis=1)
        else:
            combined_scores = (
                np_softmax(action_scores[:, 1:], dim=1) *
                np.exp(complete_scores))
            keep_idx = np.argsort(combined_scores.ravel())[-top_k:]
            for k in keep_idx:
                class_idx = k % num_classes
                proposal_idx = k // num_classes
                new_item = [
                    relative_proposals[proposal_idx,
                                       0], relative_proposals[proposal_idx, 1],
                    combined_scores[proposal_idx, class_idx],
                    regression_scores[proposal_idx, class_idx,
                                      0], regression_scores[proposal_idx,
                                                            class_idx, 1]
                ]
                if video_id not in detections[class_idx]:
                    detections[class_idx][video_id] = np.array([new_item])
                else:
                    detections[class_idx][video_id] = np.vstack(
                        [detections[class_idx][video_id], new_item])

    return detections


def perform_regression(detections):
    """Perform regression on detection results.

    Args:
        detections (list): Detection results before regression.

    Returns:
        list: Detection results after regression.
    """
    starts = detections[:, 0]
    ends = detections[:, 1]
    centers = (starts + ends) / 2
    durations = ends - starts

    new_centers = centers + durations * detections[:, 3]
    new_durations = durations * np.exp(detections[:, 4])

    new_detections = np.concatenate(
        (np.clip(new_centers - new_durations / 2, 0,
                 1)[:, None], np.clip(new_centers + new_durations / 2, 0,
                                      1)[:, None], detections[:, 2:]),
        axis=1)
    return new_detections


def temporal_nms(detections, threshold):
    """Parse the video's information.

    Args:
        detections (list): Detection results before NMS.
        threshold (float): Threshold of NMS.

    Returns:
        list: Detection results after NMS.
    """
    starts = detections[:, 0]
    ends = detections[:, 1]
    scores = detections[:, 2]

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = temporal_iou(starts[order[1:]], ends[order[1:]], starts[i],
                            ends[i])
        idxs = np.where(ious <= threshold)[0]
        order = order[idxs + 1]

    return detections[keep, :]


def eval_ap(detections, gt_by_cls, iou_range):
    """Evaluate average precisions.

    Args:
        detections (dict): Results of detections.
        gt_by_cls (dict): Information of groudtruth.
        iou_range (list): Ranges of iou.

    Returns:
        list: Average precision values of classes at ious.
    """
    ap_values = np.zeros((len(detections), len(iou_range)))

    for iou_idx, min_overlap in enumerate(iou_range):
        for class_idx in range(len(detections)):
            ap = compute_average_precision_detection(gt_by_cls[class_idx],
                                                     detections[class_idx],
                                                     [min_overlap])
            ap_values[class_idx, iou_idx] = ap

    return ap_values
