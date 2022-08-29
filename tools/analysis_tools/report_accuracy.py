# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine import load
from scipy.special import softmax

from mmaction.evaluation.functional import (get_weighted_score,
                                            mean_class_accuracy,
                                            top_k_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description='Fusing multiple scores')
    parser.add_argument(
        '--preds',
        nargs='+',
        help='list of predict result',
        default=['demo/fuse/rgb.pkl', 'demo/fuse/flow.pkl'])
    parser.add_argument(
        '--coefficients',
        nargs='+',
        type=float,
        help='coefficients of each score file',
        default=[1.0, 1.0])
    parser.add_argument(
        '--datalist',
        help='list of testing data',
        default='demo/fuse/data_list.txt')
    parser.add_argument('--apply-softmax', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.preds) == len(args.coefficients)
    data_sample_list = args.preds
    data_sample_list = [load(f) for f in data_sample_list]
    score_list = []
    for data_samples in data_sample_list:
        scores = [
            sample['pred_scores']['item'].numpy() for sample in data_samples
        ]
        score_list.append(scores)
    labels = [
        sample['gt_labels']['item'].item() for sample in data_sample_list[0]
    ]

    if args.apply_softmax:

        def apply_softmax(scores):
            return [softmax(score) for score in scores]

        score_list = [apply_softmax(scores) for scores in score_list]

    weighted_scores = get_weighted_score(score_list, args.coefficients)
    # data = open(args.datalist).readlines()
    # labels = [int(x.strip().split()[-1]) for x in data]

    mean_class_acc = mean_class_accuracy(weighted_scores, labels)
    top_1_acc, top_5_acc = top_k_accuracy(weighted_scores, labels, (1, 5))
    print(f'Mean Class Accuracy: {mean_class_acc:.04f}')
    print(f'Top 1 Accuracy: {top_1_acc:.04f}')
    print(f'Top 5 Accuracy: {top_5_acc:.04f}')


if __name__ == '__main__':
    main()
