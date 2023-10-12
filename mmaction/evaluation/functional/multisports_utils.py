# ------------------------------------------------------------------------------
# Adapted from https://github.com/MCG-NJU/MultiSports
# Original licence: Copyright (c) MCG-NJU, under the MIT License.
# ------------------------------------------------------------------------------

import math
from collections import defaultdict

import numpy as np
from mmengine.logging import MMLogger
from rich.progress import track


def area2d_voc(b):
    """Compute the areas for a set of 2D boxes."""
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def overlap2d_voc(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2."""
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2], b2[:, 2])
    ymax = np.minimum(b1[:, 3], b2[:, 3])

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d_voc(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2."""
    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d_voc(b1, b2)

    return ov / (area2d_voc(b1) + area2d_voc(b2) - ov)


def iou3d_voc(b1, b2):
    """Compute the IoU between two tubes with same temporal extent."""
    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d_voc(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d_voc(b1[:, 1:5]) + area2d_voc(b2[:, 1:5]) - ov))


def iou3dt_voc(b1, b2, spatialonly=False, temporalonly=False):
    """Compute the spatio-temporal IoU between two tubes."""
    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0])

    tube1 = b1[int(np.where(
        b1[:, 0] == tmin)[0]):int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(
        b2[:, 0] == tmin)[0]):int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    if temporalonly:
        return temporal_inter / temporal_union
    return iou3d_voc(tube1, tube2) * (1. if spatialonly else temporal_inter /
                                      temporal_union)


def pr_to_ap_voc(pr):
    precision = pr[:, 0]
    recall = pr[:, 1]
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision


def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets scored tubelets are numpy
    array with 4K+1 columns, last one being the score return the indices of the
    tubelets to keep."""

    # If there are no detections, return an empty list
    if len(dets) == 0:
        return dets
    if top_k is None:
        top_k = len(dets)

    K = int((dets.shape[1] - 1) / 4)

    # Coordinates of bounding boxes
    x1 = [dets[:, 4 * k] for k in range(K)]
    y1 = [dets[:, 4 * k + 1] for k in range(K)]
    x2 = [dets[:, 4 * k + 2] for k in range(K)]
    y2 = [dets[:, 4 * k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, -1]
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1
    counter = 0

    while order.size > 0:
        i = order[0]
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][order[1:]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][order[1:]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][order[1:]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][order[1:]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([
            inter_area[k] / (area[k][order[1:]] + area[k][i] - inter_area[k])
            for k in range(K)
        ])
        index = np.where(ious > overlapThresh * K)[0]
        weight[order[index + 1]] = 1 - ious[index]

        index2 = np.where(ious <= overlapThresh * K)[0]
        order = order[index2 + 1]

    dets[:, -1] = dets[:, -1] * weight

    new_scores = dets[:, -1]
    new_order = np.argsort(new_scores)[::-1]
    dets = dets[new_order, :]

    return dets[:top_k, :]


class Dataset():

    def __init__(self, anno, frm_alldets) -> None:
        self.anno = anno
        self.video_list = self.anno['test_videos'][0]
        self.nframes = self.anno['nframes']
        self.labels = self.anno['labels']
        self.frm_alldets = frm_alldets

    def get_vid_dets(self):
        self.vid_frm_det = defaultdict(list)
        for frm_det in self.frm_alldets:
            vid_idx = int(frm_det[0])
            vid_name = self.video_list[vid_idx]
            self.vid_frm_det[vid_name].append(frm_det)

        self.vid_det = dict()
        for vid_name, vid_frm_dets in self.vid_frm_det.items():
            self.vid_det[vid_name] = dict()
            for frm_idx in range(1, self.nframes[vid_name] + 1):
                self.vid_det[vid_name][frm_idx] = dict()
                for label_idx in range(len(self.labels)):
                    self.vid_det[vid_name][frm_idx][label_idx] = np.empty(
                        shape=(0, 5))
            for frm_dets in vid_frm_dets:
                frm_idx = int(frm_dets[1])
                label_idx = int(frm_dets[2])
                det = [*frm_dets[-4:], frm_det[3]]
                det = np.array(det)[None, :]

                self.vid_det[vid_name][frm_idx][label_idx] = np.concatenate(
                    [self.vid_det[vid_name][frm_idx][label_idx], det])

        return self.vid_det


def link_tubes(anno, frm_dets, K=1, len_thre=15):

    dataset = Dataset(anno, frm_dets)
    vlist = dataset.video_list
    total_VDets = dataset.get_vid_dets()

    total_video_tubes = {label: [] for label in range(len(dataset.labels))}
    for v in track(vlist, description='linking tubes...'):

        RES = {}
        if v not in total_VDets:
            continue
        VDets = total_VDets[v]
        for ilabel in range(len(dataset.labels)):
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)

            # calculate average scores of tubelets in tubes

            def tubescore(tt):
                return np.mean(
                    np.array([tt[i][1][-1] for i in range(len(tt))]))

            for frame in range(1, dataset.nframes[v] + 2 - K):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored # noqa: E501
                ltubelets = np.array(
                    VDets[frame][ilabel]
                )  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score  # noqa: E501

                ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)

                # just start new tubes
                if frame == 1:
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(1, ltubelets[i, :])])
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    ious = []
                    offset = frame - last_frame
                    if offset < K:
                        nov = K - offset
                        ious = sum([
                            iou2d_voc(
                                ltubelets[:, 4 * iov:4 * iov + 4],
                                last_tubelet[4 * (iov + offset):4 *
                                             (iov + offset + 1)])
                            for iov in range(nov)
                        ]) / float(nov)
                    else:
                        ious = iou2d_voc(ltubelets[:, :4],
                                         last_tubelet[4 * K - 4:4 * K])

                    valid = np.where(ious >= 0.5)[0]

                    if valid.size > 0:
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= K:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::
                                   -1]:  # process in reverse order to delete them with the right index why --++-- # noqa: E501
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)

                # just start new tubes
                if score < 0.005:
                    continue

                beginframe = t[0][0]
                endframe = t[-1][0] + K - 1
                length = endframe + 1 - beginframe

                # delete tubes with short duraton
                if length < len_thre:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(K):
                        out[frame - beginframe + k,
                            1:5] += box[4 * k:4 * k + 4]
                        out[frame - beginframe + k,
                            -1] += box[-1]  # single frame confidence
                        n_per_frame[frame - beginframe + k, 0] += 1
                out[:, 1:] /= n_per_frame
                output.append([out, score])
                # out: [num_frames, (frame idx, x1, y1, x2, y2, score)]

            RES[ilabel] = output
            if output:
                for tube, tube_score in output:
                    video_tube_res = tuple([v, tube_score, tube])
                    total_video_tubes[ilabel].append(video_tube_res)
    return total_video_tubes


def frameAP(GT, alldets, thr, print_info=True):
    logger = MMLogger.get_current_instance()
    vlist = GT['test_videos'][0]

    results = {}
    for ilabel, label in enumerate(GT['labels']):
        # detections of this class
        if label in [
                'aerobic kick jump', 'aerobic off axis jump',
                'aerobic butterfly jump', 'aerobic balance turn',
                'basketball save', 'basketball jump ball'
        ]:
            if print_info:
                logger.info('do not evaluate {}'.format(label))
            continue
        # det format: <video_index><frame_number><label_index><score><x1><y1><x2><y2> # noqa: E501
        detections = alldets[alldets[:, 2] == ilabel, :]

        # load ground-truth of this class
        gt = {}
        for iv, v in enumerate(vlist):
            tubes = GT['gttubes'][v]

            if ilabel not in tubes:
                continue

            for tube in tubes[ilabel]:
                for i in range(tube.shape[0]):
                    k = (iv, int(tube[i, 0]))  # k -> (video_idx, frame_idx)
                    if k not in gt:
                        gt[k] = []
                    gt[k].append(tube[i, 1:5].tolist())

        for k in gt:
            gt[k] = np.array(gt[k])

        # pr will be an array containing precision-recall values
        pr = np.empty((detections.shape[0], 2),
                      dtype=np.float64)  # precision,recall
        gt_num = sum([g.shape[0] for g in gt.values()])
        if gt_num == 0:
            if print_info:
                logger.info('no such label', ilabel, label)
            continue
        fp = 0  # false positives
        tp = 0  # true positives

        is_gt_box_detected = {}
        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]
            ispositive = False

            if k in gt:
                # match gt_box according to the iou
                if k not in is_gt_box_detected:
                    is_gt_box_detected[k] = np.zeros(
                        gt[k].shape[0], dtype=bool)
                ious = iou2d_voc(gt[k], box)
                amax = np.argmax(ious)

                if ious[amax] >= thr:
                    if not is_gt_box_detected[k][amax]:
                        ispositive = True
                        is_gt_box_detected[k][amax] = True

            if ispositive:
                tp += 1
            else:
                fp += 1
            pr[i, 0] = float(tp) / float(tp + fp)
            pr[i, 1] = float(tp) / float(gt_num)

        results[label] = pr

    # display results
    ap = 100 * np.array([pr_to_ap_voc(results[label]) for label in results])
    class_result = {}
    for label in results:
        class_result[label] = pr_to_ap_voc(results[label]) * 100
    frameap_result = np.mean(ap)
    if print_info:
        logger.info('frameAP_{}\n'.format(thr))
        for label in class_result:
            logger.info('{:20s} {:8.2f}'.format(label, class_result[label]))
        logger.info('{:20s} {:8.2f}'.format('mAP', frameap_result))
    return frameap_result


def videoAP(GT, alldets, thr, print_info=True):
    logger = MMLogger.get_current_instance()
    vlist = GT['test_videos'][0]

    res = {}
    for ilabel in range(len(GT['labels'])):
        if GT['labels'][ilabel] in [
                'aerobic kick jump', 'aerobic off axis jump',
                'aerobic butterfly jump', 'aerobic balance turn',
                'basketball save', 'basketball jump ball'
        ]:
            if print_info:
                logger.info('do not evaluate{}'.format(GT['labels'][ilabel]))
            continue
        detections = alldets[ilabel]
        # load ground-truth
        gt = {}
        for v in vlist:
            tubes = GT['gttubes'][v]

            if ilabel not in tubes:
                continue

            gt[v] = tubes[ilabel]

            if len(gt[v]) == 0:
                del gt[v]

        # precision,recall
        pr = np.empty((len(detections), 2), dtype=np.float64)

        gt_num = sum([len(g) for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives
        if gt_num == 0:
            if print_info:
                logger.info('no such label', ilabel, GT['labels'][ilabel])
            continue
        is_gt_box_detected = {}
        for i, j in enumerate(
                np.argsort(-np.array([dd[1] for dd in detections]))):
            v, score, tube = detections[j]
            ispositive = False
            if v in gt:
                if v not in is_gt_box_detected:
                    is_gt_box_detected[v] = np.zeros(len(gt[v]), dtype=bool)
                ious = [iou3dt_voc(g, tube) for g in gt[v]]
                amax = np.argmax(ious)
                if ious[amax] >= thr:
                    if not is_gt_box_detected[v][amax]:
                        ispositive = True
                        is_gt_box_detected[v][amax] = True

            if ispositive:
                tp += 1
            else:
                fp += 1

            pr[i, 0] = float(tp) / float(tp + fp)
            pr[i, 1] = float(tp) / float(gt_num)
        res[GT['labels'][ilabel]] = pr

    # display results
    ap = 100 * np.array([pr_to_ap_voc(res[label]) for label in res])
    videoap_result = np.mean(ap)
    class_result = {}
    for label in res:
        class_result[label] = pr_to_ap_voc(res[label]) * 100
    if print_info:
        logger.info('VideoAP_{}\n'.format(thr))
        for label in class_result:
            logger.info('{:20s} {:8.2f}'.format(label, class_result[label]))
        logger.info('{:20s} {:8.2f}'.format('mAP', videoap_result))
    return videoap_result


def videoAP_all(groundtruth, detections):
    high_ap = 0
    for i in range(10):
        thr = 0.5 + 0.05 * i
        high_ap += videoAP(groundtruth, detections, thr, print_info=False)
    high_ap = high_ap / 10.0

    low_ap = 0
    for i in range(9):
        thr = 0.05 + 0.05 * i
        low_ap += videoAP(groundtruth, detections, thr, print_info=False)
    low_ap = low_ap / 9.0

    all_ap = 0
    for i in range(9):
        thr = 0.1 + 0.1 * i
        all_ap += videoAP(groundtruth, detections, thr, print_info=False)
    all_ap = all_ap / 9.0

    map = {
        'v_map_0.05:0.45': round(low_ap, 4),
        'v_map_0.10:0.90': round(all_ap, 4),
        'v_map_0.50:0.95': round(high_ap, 4),
    }
    return map


def videoAP_error(GT, alldets, thr):

    vlist = GT['test_videos'][0]

    th_s = math.sqrt(thr)
    th_t = math.sqrt(thr)

    print('th is', thr)
    print('th_s is', th_s)
    print('th_t is', th_t)

    res = {}
    dupgt = {}
    for v in vlist:
        dupgt[v] = GT['gttubes'][v]
    # compute video error for every class
    for ilabel in range(len(GT['labels'])):
        if GT['labels'][ilabel] in [
                'aerobic kick jump', 'aerobic off axis jump',
                'aerobic butterfly jump', 'aerobic balance turn',
                'basketball save', 'basketball jump ball'
        ]:
            print('do not evaluate {}'.format(GT['labels'][ilabel]))
            continue
        detections = alldets[ilabel]

        pr = np.zeros((len(detections), 11), dtype=np.float32)

        gt_num = 0
        for v in dupgt:
            if ilabel in dupgt[v]:
                gt_num = gt_num + len(dupgt[v][ilabel])
        fp = 0  # false positives
        tp = 0  # true positives
        ER = 0  # repeat error repeat predict for the same instance
        EN = 0  # extra error
        EL = 0  # localization errors
        EC = 0  # classification error
        ET = 0  # timing error
        ErrCT = 0  # cls + time
        ECL = 0  # cls + loc
        ETL = 0  # time + loc
        ECTL = 0  # cls + time + loc

        is_gt_box_detected = {}
        for i, j in enumerate(
                np.argsort(-np.array([dd[1] for dd in detections]))):
            v, score, tube = detections[j]
            ispositive = False
            end = False
            if ilabel in dupgt[v]:
                if v not in is_gt_box_detected:
                    is_gt_box_detected[v] = np.zeros(
                        len(dupgt[v][ilabel]), dtype=bool)
                ious = [iou3dt_voc(g, tube) for g in dupgt[v][ilabel]]
                amax = np.argmax(ious)
                if ious[amax] >= thr:
                    if not is_gt_box_detected[v][amax]:
                        ispositive = True
                        is_gt_box_detected[v][amax] = True
                    else:
                        ER += 1
                    end = True
            if end is False:
                ious = []
                for ll in dupgt[v]:
                    if ll == ilabel:
                        continue
                    for g in dupgt[v][ll]:
                        ious.append(iou3dt_voc(g, tube))
                if ious != []:
                    amax = np.argmax(ious)
                    if ious[amax] >= thr:
                        EC += 1
                        end = True
            if end is False:
                all_gt = []
                ious = []
                for ll in dupgt[v]:
                    for g in dupgt[v][ll]:
                        all_gt.append((ll, g))
                        ious.append(iou3dt_voc(g, tube))
                amax = np.argmax(ious)
                assert (ious[amax] < thr)
                if ious[amax] > 0:
                    t_iou = iou3dt_voc(
                        all_gt[amax][1], tube, temporalonly=True)
                    s_iou = iou3dt_voc(all_gt[amax][1], tube, spatialonly=True)
                    if all_gt[amax][0] == ilabel:
                        assert (t_iou < th_t or s_iou < th_s)
                        if t_iou >= th_t:
                            EL += 1
                            end = True
                        elif s_iou >= th_s:
                            ET += 1
                            end = True
                        else:
                            ETL += 1
                            end = True
                    else:
                        assert (t_iou < th_t or s_iou < th_s)
                        if t_iou >= th_t:
                            ECL += 1
                            end = True
                        elif s_iou >= th_s:
                            ErrCT += 1
                            end = True
                        else:
                            ECTL += 1
                            end = True
                else:
                    EN += 1
                    end = True
            assert (end is True)
            if ispositive:
                tp += 1
                # fn -= 1
            else:
                fp += 1
            assert (fp == (ER + EN + EL + EC + ET + ErrCT + ECL + ETL + ECTL))
            pr[i, 0] = max(float(tp) / float(tp + fp), 0.)
            pr[i, 1] = max(float(tp) / float(gt_num), 0.)
            pr[i, 2] = max(float(ER) / float(tp + fp), 0.)
            pr[i, 3] = max(float(EN) / float(tp + fp), 0.)
            pr[i, 4] = max(float(EL) / float(tp + fp), 0.)
            pr[i, 5] = max(float(EC) / float(tp + fp), 0.)
            pr[i, 6] = max(float(ET) / float(tp + fp), 0.)
            pr[i, 7] = max(float(ErrCT) / float(tp + fp), 0.)
            pr[i, 8] = max(float(ECL) / float(tp + fp), 0.)
            pr[i, 9] = max(float(ETL) / float(tp + fp), 0.)
            pr[i, 10] = max(float(ECTL) / float(tp + fp), 0.)

        res[GT['labels'][ilabel]] = pr

    # display results
    AP = 100 * np.array([pr_to_ap_voc(res[label][:, [0, 1]]) for label in res])
    othersap = [
        100 * np.array([pr_to_ap_voc(res[label][:, [j, 1]]) for label in res])
        for j in range(2, 11)
    ]

    ER = othersap[0]
    EN = othersap[1]
    EL = othersap[2]
    EC = othersap[3]
    ET = othersap[4]
    ErrCT = othersap[5]
    ECL = othersap[6]
    ETL = othersap[7]
    ECTL = othersap[8]
    # missed detections = 1-recalll
    EM = []
    for label in res:
        if res[label].shape[0] != 0:
            EM.append(100 - 100 * res[label][-1, 1])
        else:
            EM.append(100)
    EM = np.array(EM)

    LIST = [AP, ER, EN, EL, EC, ET, ErrCT, ECL, ETL, ECTL, EM]

    print('Error Analysis')

    print('')
    print(
        '{:20s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}'  # noqa: E501
        .format('label', '  AP ', '  Repeat ', ' Extra ', ' Loc. ', ' Cls. ',
                ' Time ', ' Cls.+Time ', ' Cls.+Loc. ', ' Time+Loc. ',
                ' C+T+L ', ' missed '))
    print('')
    for il, label in enumerate(res):
        print('{:20s} '.format(label) +
              ' '.join(['{:8.2f}'.format(L[il]) for L in LIST]))
    print('')
    print('{:20s} '.format('mean') +
          ' '.join(['{:8.2f}'.format(np.mean(L)) for L in LIST]))
    print('')
