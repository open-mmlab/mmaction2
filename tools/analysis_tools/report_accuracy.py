# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
from mmengine import load
from scipy.special import softmax

from mmaction.evaluation.functional import (get_weighted_score,
                                            mean_class_accuracy,
                                            mmit_mean_average_precision,
                                            top_k_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description='Fusing multiple scores')
    parser.add_argument(
        '--preds',
        nargs='+',
        help='list of predict result',
        default=['demo/fuse/joint.pkl', 'demo/fuse/bone.pkl'])
    parser.add_argument(
        '--coefficients',
        nargs='+',
        type=float,
        help='coefficients of each score file',
        default=[1.0, 1.0])
    parser.add_argument('--apply-softmax', action='store_true')
    parser.add_argument(
        '--multi-label',
        action='store_true',
        help='whether the task is multi label classification')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.preds) == len(args.coefficients)
    data_sample_list = [load(f) for f in args.preds]
    score_list = []
    for data_samples in data_sample_list:
        scores = [sample['pred_score'].numpy() for sample in data_samples]
        score_list.append(scores)

    if args.multi_label:
        labels = [sample['gt_label'] for sample in data_sample_list[0]]
    else:
        labels = [sample['gt_label'].item() for sample in data_sample_list[0]]

    if args.apply_softmax:

        def apply_softmax(scores):
            return [softmax(score) for score in scores]

        score_list = [apply_softmax(scores) for scores in score_list]

    weighted_scores = get_weighted_score(score_list, args.coefficients)
    if args.multi_label:
        mean_avg_prec = mmit_mean_average_precision(
            np.array(weighted_scores), np.stack([t.numpy() for t in labels]))
        print(f'MMit Average Precision: {mean_avg_prec:.04f}')
    else:
        mean_class_acc = mean_class_accuracy(weighted_scores, labels)
        top_1_acc, top_5_acc = top_k_accuracy(weighted_scores, labels, (1, 5))
        print(f'Mean Class Accuracy: {mean_class_acc:.04f}')
        print(f'Top 1 Accuracy: {top_1_acc:.04f}')
        print(f'Top 5 Accuracy: {top_5_acc:.04f}')


if __name__ == '__main__':
    main()
