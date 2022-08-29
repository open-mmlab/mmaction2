import os
import pickle
import json

import cv2 as cv
import numpy as np
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_ddp, build_dp


def coco_eval(args):
    _dataset = args.dataset
    data_root = f'/home/jaeguk/workspace/data/{_dataset}/'
    cfg = Config.fromfile(args.det_config)
    cfg.data.test.ann_file = os.path.join(
        data_root, 'annotations', 'instances_valid.json'
    )
    cfg.data.test.img_prefix = os.path.join(
        data_root, 'frames'
    )
    cfg.data.test.test_mode = True
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False
    )
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.det_checkpoint, map_location='cpu')
    model.CLASSES = ('person')
    model = build_dp(model, 'cuda', device_ids=[0])
    show = False
    if show:
        show_dir = f'/home/jaeguk/workspace/data/{_dataset}/test/results/'
        os.makedirs(show_dir, exist_ok=True)
        outputs = single_gpu_test(model, data_loader, True, show_dir, 0.5)
    else:
        outputs = single_gpu_test(model, data_loader, False)
    eval_kwargs = cfg.get('evaluation', {}).copy()
    eval_kwargs.update(dict(metric='bbox', classwise=True))
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)

def make_proposals(args):
    frame_paths = []
    data_root = f'/home/jaeguk/workspace/data/{args.dataset}'
    img_root= os.path.join(data_root, 'frames')
    json_file = os.path.join(data_root, 'annotations', args.json_file)
    with open(json_file) as f:
        json_info = json.load(f)
    for image in json_info['images']:
        frame_paths.append(image['file_name'])

    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    print('Performing Human Detection for each frame')
    results = {}
    json_name = args.json_file.split('.')[0]
    with open(os.path.join(data_root, 'annotations',
         f'{args.dataset}_dense_proposals_{json_name}.pkl'), 'wb') as fb:
        prog_bar = mmcv.ProgressBar(len(frame_paths))
        for frame_path in frame_paths:
            img_name = frame_path.split('/')[-1]
            vid = frame_path.replace(f'/{img_name}', '')
            frame_num = img_name.split('.')[0]
            cv_img = cv.imread(os.path.join(img_root, frame_path))
            h, w, c = cv_img.shape
            divider = np.array([w, h, w, h])
            result = inference_detector(model, os.path.join(img_root, frame_path))
            result = result[0][result[0][:, 4] >= args.det_score_thr]
            if len(result) < 1:
                prog_bar.update()
                continue
            result[:, :4] = result[:, :4] / divider
            results[f'{vid},{frame_num}'] = result
            prog_bar.update()
        pickle.dump(results, fb)

def check_proposals(args):
    json_name = args.json_file.split('.')[0]
    pkl_name = f'{args.dataset}_dense_proposals_{json_name}.pkl'
    img_root = f'/home/jaeguk/workspace/data/{args.dataset}/frames'
    annot_root = f'/home/jaeguk/workspace/data/{args.dataset}/annotations'
    save_root = f'/home/jaeguk/workspace/data/{args.dataset}/validation/{pkl_name}/'
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(annot_root, pkl_name), 'rb') as fb:
        proposals = pickle.load(fb, encoding='latin1')
        prog_bar = mmcv.ProgressBar(len(proposals))
        for key in proposals.keys():
            proposal = proposals[key]
            vid, frame = key.split(',')
            if not os.path.exists(os.path.join(save_root, vid)):
                os.makedirs(os.path.join(save_root, vid))
            img_name = os.path.join(img_root, vid, f'{frame}.{args.ext}')
            img = cv.imread(img_name)
            h, w, c = img.shape
            for bbox in proposal:
                x1,y1,x2,y2,p = bbox
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv.rectangle(img, p1, p2, (255,0,0), 2)
            cv.imwrite(os.path.join(save_root, vid, f'{frame}.{args.ext}'), img)
            prog_bar.update()

if __name__ == '__main__':
    class args:
        det_config = '/home/jaeguk/workspace/mmaction2/demo/' \
                     'faster_rcnn_r50_fpn_2x_coco.py'
        det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/' \
                         'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/' \
                         'faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_' \
                         '20200504_210434-a5d8aa15.pth'
        #det_checkpoint = '/home/jaeguk/workspace/mmaction2/work_dirs/' \
        #                 'faster_rcnn_r50_fpn_2x_coco/best_e19_mAP46.pth'
        det_score_thr = 0.75
        device = 'cuda:1'
        get_eval_score = False
        make_proposals_bytes = True
        check_proposals = False
        dataset = 'JHMDB'
        ext = 'jpg'
        json_file = 'instances_valid.json'

    for k, v in vars(args).items():
        if k[0] != '_':
            print(f'{k}: {v}')

    if args.get_eval_score:
        coco_eval(args)
    if args.make_proposals_bytes:
        make_proposals(args)
    if args.check_proposals:
        check_proposals(args)
