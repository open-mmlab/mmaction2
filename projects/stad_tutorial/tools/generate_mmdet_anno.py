# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import defaultdict

from mmengine import dump, load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'stad_anno', help='spatiotemporal action detection anno path')
    parser.add_argument('det_path', help='output detection anno path')
    args = parser.parse_args()
    return args


def generate_mmdet_coco_anno(args):
    ori_anno = load(args.stad_anno)
    train_videos = ori_anno['train_videos']
    val_videos = ori_anno['test_videos']
    videos = {'train': train_videos, 'val': val_videos}
    for split in ['train', 'val']:
        img_id = 0
        bbox_id = 0
        img_list = []
        anno_list = []
        for vid in videos[split][0]:
            vid_tubes = ori_anno['gttubes'][vid]
            height, width = ori_anno['resolution'][vid]
            frm2bbox = defaultdict(list)
            for label_idx, tube_list in vid_tubes.items():
                for tube in tube_list:
                    for frm_anno in tube:
                        frm_idx, bbox = frm_anno[0], frm_anno[1:]
                        frm2bbox[frm_idx].append({'label': 0, 'bbox': bbox})
            for frm_idx, frm_bboxes in frm2bbox.items():
                img_path = f'{vid}/{int(frm_idx):05d}.jpg'
                img_instance = {
                    'file_name': img_path,
                    'height': height,
                    'width': width,
                    'id': img_id
                }
                img_list.append(img_instance)

                for bbox_info in frm_bboxes:
                    label = bbox_info['label']
                    x1, y1, x2, y2 = bbox_info['bbox']
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    anno_instance = {
                        'area': bbox[2] * bbox[3],
                        'image_id': img_id,
                        'bbox': bbox,
                        'category_id': label,
                        'iscrowd': 0,
                        'id': bbox_id
                    }
                    anno_list.append(anno_instance)
                    bbox_id += 1
                img_id += 1
        total_anno = {
            'images': img_list,
            'annotations': anno_list,
            'categories': [{
                'id': 0,
                'name': 'person'
            }],
        }
        dump(total_anno, args.det_path[:-5] + f'_{split}' + args.det_path[-5:])


if __name__ == '__main__':
    args = parse_args()
    generate_mmdet_coco_anno(args)
