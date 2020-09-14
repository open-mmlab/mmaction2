import argparse
import os
import os.path as osp

import mmcv
import numpy as np

from mmaction.core import ANETdetection

args = None


def cuhk17_top1():
    global args
    if not osp.exists('cuhk_anet17_pred.json'):
        os.system('wget https://openmmlab.oss-accelerate.aliyuncs.com/'
                  'mmaction/localization/cuhk_anet17_pred.json')
    proposal = mmcv.load(args.proposal)
    results = proposal['results']
    cuhk = mmcv.load('cuhk_anet17_pred.json')
    cuhk = cuhk['results']

    def get_topk(preds, k):
        preds.sort(key=lambda x: x['score'])
        return preds[-k:]

    for k, v in results.items():
        action_pred = cuhk[k]
        top1 = get_topk(action_pred, 1)
        lb = top1[0]['label']
        newv = []
        for item in v:
            x = dict(label=lb)
            x.update(item)
            newv.append(x)
        results[k] = newv
    proposal['results'] = results
    mmcv.dump(proposal, args.det_output)


cls_funcs = {'cuhk17_top1': cuhk17_top1}


def parse_args():
    parser = argparse.ArgumentParser(description='Report detection mAP for'
                                     'ActivityNet proposal file')
    parser.add_argument('--proposal', type=str, help='proposal file')
    parser.add_argument(
        '--gt',
        type=str,
        default='data/ActivityNet/'
        'anet_anno_val.json',
        help='groundtruth file')
    parser.add_argument(
        '--cls',
        type=str,
        default='cuhk17_top1',
        choices=['cuhk17_top1'],
        help='the way to assign label for each '
        'proposal')
    parser.add_argument(
        '--det-output',
        type=str,
        default='det_result.json',
        help='the path to store detection results')
    args = parser.parse_args()
    return args


def main():
    global args, cls_funcs
    args = parse_args()
    func = cls_funcs[args.cls]
    func()
    anet_detection = ANETdetection(
        args.gt,
        args.det_output,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        verbose=True)
    mAP, average_mAP = anet_detection.evaluate()
    print('[RESULTS] Performance on ActivityNet detection task.')
    print(f'mAP: {mAP}')
    print(f'Average-mAP: {average_mAP}')


if __name__ == '__main__':
    main()
