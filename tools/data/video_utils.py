import pickle
import os
import json
import cv2 as cv
import numpy as np
import csv
from random import shuffle


def annot2json():
    '''
    convert pkl pre annotated bounding box information to coco style json file
    '''

    dataset_name = 'JHMDB'
    annot_pkl = f'/home/jaeguk/workspace/data/{dataset_name}/_annotations/JHMDB-GT.pkl'
    frame_dir = f'/home/jaeguk/workspace/data/{dataset_name}/frames/'
    train_json = f'/home/jaeguk/workspace/data/{dataset_name}/annotations/instances_train.json'
    valid_json = f'/home/jaeguk/workspace/data/{dataset_name}/annotations/instances_valid.json'

    train_categories = [{"id": 1, "name": "person"}]
    train_images = []
    train_annotations = []
    valid_categories = train_categories
    valid_images = []
    valid_annotations = []

    train_img_id = 0
    train_annot_id = 0
    valid_img_id = 0
    valid_annot_id = 0

    train_img_ids = {}
    valid_img_ids = {}

    ext = 'png' if dataset_name in ['JHMDB'] else 'jpg'

    with open(annot_pkl, 'rb') as f:
        info = pickle.load(f, encoding='latin1')
        gttubes = info['gttubes']  # {Label: [array(f, x1, y1, x2, y2)]}
        resolutions = info['resolution']
        train_lists = info['train_videos']

        for action in os.listdir(frame_dir):
            if action[0] =='.':
                # continue .appledouble
                continue
            clip_dir = os.listdir(os.path.join(frame_dir, action))
            shuffle(clip_dir)
            for idx, clip in enumerate(clip_dir):
                if clip[0] == ".":
                    continue
                if idx < len(os.listdir(os.path.join(frame_dir, action))) - 30:
                    clip_status = 'train'
                else:
                    clip_status = 'val'
                vid = f'{action}/{clip}'
                try:
                    h, w = resolutions[vid]
                except:
                    h = None
                    w = None
                    continue
                annots = gttubes[vid]
                if len(annots.keys()) > 1:
                    print('Warning: annots have more than 1 key: ', annots)
                label = list(annots.keys())[0]
                instances = annots[label]
                for bboxes in instances:
                    for bbox in bboxes:
                        frame, x1, y1, x2, y2 = bbox
                        x1 = float(x1)
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        file_name = f'{vid}/{int(frame):05}.{ext}'
                        if h == None or w == None:
                            img = cv.imread(os.path.join(frame_dir, file_name))
                            if img is None:
                                raise Exception(os.path.join(frame_dir, file_name),
                                                ' is not exist!')
                            h = img.shape[0]
                            w = img.shape[1]
                            resolutions[vid] = (h, w)
                        if clip_status == 'train':
                            if file_name not in train_img_ids.keys():
                                train_img_ids[file_name] = train_annot_id
                                nimg = {"id": train_img_ids[file_name],
                                        "file_name": file_name,
                                        "height": h, "width": w}
                                train_images.append(nimg)
                                train_annot_id += 1
                            nannot = {
                                "id": train_annot_id,
                                "image_id": train_img_ids[file_name],
                                "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0
                            }
                            train_annotations.append(nannot)
                            train_annot_id += 1
                        else:
                            if file_name not in valid_img_ids.keys():
                                valid_img_ids[file_name] = valid_annot_id
                                nimg = {"id": valid_img_ids[file_name],
                                        "file_name": file_name,
                                        "height": h, "width": w}
                                valid_images.append(nimg)
                                valid_annot_id += 1
                            nannot = {
                                "id": valid_annot_id,
                                "image_id": valid_img_ids[file_name],
                                "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0
                            }
                            valid_annotations.append(nannot)
                            valid_annot_id += 1

    with open(train_json, 'w') as tjf:
        info = {'categories': train_categories,
                'images': train_images, 'annotations': train_annotations}
        json.dump(info, tjf, indent=4)
    with open(valid_json, 'w') as vjf:
        info = {'categories': valid_categories,
                'images': valid_images, 'annotations': valid_annotations}
        json.dump(info, vjf, indent=4)

def json2csv(name):
    annot_root = '/home/jaeguk/workspace/data/JHMDB/annotations/'
    json_file = os.path.join(annot_root, f'instances_{name}.json')
    csv_file = os.path.join(annot_root, f'JHMDB_{name}.csv')
    pbtxt_file = os.path.join(annot_root, 'JHMDB_actionlist.pbtxt')
    action_label = get_action_label(pbtxt_file)
    with open(json_file) as jf, open(csv_file, 'w') as cf:
        idx = 0
        wr = csv.writer(cf)
        info = json.load(jf)
        for image in info['images']:
            name = image['file_name']
            frame = name.split('/')[-1]
            vid = name.replace(f'/{frame}', '')
            action = vid.split('/')[0]
            try:
                action_id = action_label[action]
            except:
                breakpoint()
            frame_num = int(frame.split('.')[0])
            img_id = image['id']
            height = image['height']
            width = image['width']
            for annot in info['annotations']:
                if annot['image_id'] == img_id:
                    x1, y1, w, h = annot['bbox']
                    x2 = x1 + w
                    y2 = y1 + h
                    x1 = round(x1 / width, 3)
                    x2 = round(x2 / width, 3)
                    y1 = round(y1 / height, 3)
                    y2 = round(y2 / height, 3)
                    wr.writerow([vid,frame_num,x1,y1,x2,y2,action_id,idx])
                    idx += 1

def get_action_label(pbtxt_file):
    action_label = {}
    with open(pbtxt_file) as pf:
        lines = pf.readlines()
        buf = []
        for line in lines:
            if 'name:' in line:
                action = line.split('\n')[0].split('name: ')[1]
                action = action.split('"')[1]
                assert len(buf) == 0
                buf.append(action)
            if 'id:' in line:
                action_id = line.split('\n')[0].split('id: ')[1]
                assert len(buf) == 1
                buf.append(action_id)
                action_label[buf[0]] = buf[1]
                buf = []

    return action_label

def validate_csv():
    annot_root = '/home/jaeguk/workspace/data/JHMDB/annotations/'
    img_root = '/home/jaeguk/workspace/data/JHMDB/frames/'
    save_root = '/home/jaeguk/workspace/data/JHMDB/validation/csv-sampled/'
    csv_file = os.path.join(annot_root, 'JHMDB_valid_20.csv')
    with open(csv_file) as cf:
        lines = cf.readlines()
        for line in lines:
            vid, frame, x1, y1, x2, y2, action_id, idx = line.split(',')
            x1 = int(float(x1) * 320)
            x2 = int(float(x2) * 320)
            y1 = int(float(y1) * 240)
            y2 = int(float(y2) * 240)
            image = os.path.join(img_root, vid, f'{int(frame):05}.png')
            cv_image = cv.imread(image)
            cv.rectangle(cv_image, (x1, y1), (x2, y2), (0,255,0), 1)
            if not os.path.exists(os.path.join(save_root, vid)):
                os.makedirs(os.path.join(save_root, vid))
            cv.imwrite(os.path.join(save_root, vid, f'{int(frame):05}.png'), cv_image)

def validate_json():
    annot_root = '/home/jaeguk/workspace/data/JHMDB/annotations/'
    img_root = '/home/jaeguk/workspace/data/JHMDB/frames/'
    save_root = '/home/jaeguk/workspace/data/JHMDB/validation/json/'
    json_file = os.path.join(annot_root, 'instances_valid.json')
    with open(json_file) as jf:
        info = json.load(jf)
        for image in info['images']:
            cv_img = cv.imread(os.path.join(img_root, image['file_name']))
            for annot in info['annotations']:
                if annot['image_id'] == image['id']:
                    x, y, w, h = annot['bbox']
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)
                    cv.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    save_dir = [save_root] + image['file_name'].split('/')[:-1]
                    save_dir = os.path.join(*save_dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cv.imwrite(os.path.join(save_root, image['file_name']), cv_img)

def get_pbtxt():
    video_root = '/home/jaeguk/workspace/data/JHMDB/'
    pb_txt = os.path.join(video_root, 'annotations', 'JHMDB_actionlist.pbtxt')
    with open(pb_txt, 'w') as f:
        actions = os.listdir(os.path.join(video_root, 'frames'))
        label_id = 1
        for action in actions:
            if action[0] == '.':
                continue
            f.write('item {\n')
            f.write(f'  name: "{action}"\n')
            f.write(f'  id: {label_id}\n')
            label_id += 1
            f.write('}\n')

def print_frame_len(dataset):
    video_root = f'/home/jaeguk/workspace/data/{dataset}/frames'
    len_dict = {}
    for action in os.listdir(video_root):
        if action[0] == '.':
            continue
        if dataset in ['ava']:
            frame_len = len(os.listdir(os.path.join(video_root, action)))
            len_dict[action] = frame_len
        else:
            for vid in os.listdir(os.path.join(video_root, action)):
                if vid[0] == '.':
                    continue
                frame_len = len(os.listdir(os.path.join(video_root, action, vid)))
                len_dict[vid] = frame_len
    statics = len_dict.values()
    statics = np.asarray(list(statics))
    print(f'{dataset} Dataset Statistics')
    print('min frame length: ', min(statics))
    print('average frame length', int(statics.mean()))

    return len_dict

def sample_from_json(name):
    annot_root = '/home/jaeguk/workspace/data/JHMDB/annotations/'
    pbtxt = os.path.join(annot_root, 'JHMDB_actionlist.pbtxt')
    action_label = get_action_label(pbtxt)
    frame_len_dict = print_frame_len('JHMDB')

    proposal_pkl = f'JHMDB_dense_proposals_instances_{name}.pkl'
    with open(os.path.join(annot_root, proposal_pkl), 'rb') as fb:
        proposals = pickle.load(fb, encoding='latin1')

    action_thr = 5 if name == 'train' else 2
    json_file = os.path.join(annot_root, f'instances_{name}.json')
    new_json_file = os.path.join(annot_root,
        f'instances_{name}_{action_thr * len(action_label)}.json')
    with open(json_file) as jf:
        info = json.load(jf)
        _images = info['images']
        _annotations = info['annotations']
        _categories = info['categories']

        shuffle(_images)
        shuffle(_annotations)

    action_cnt = {}
    for action in action_label.keys():
        action_cnt[action] = 0

    images = []
    image_ids = []
    for image in _images:
        action = image['file_name'].split('/')[0]
        vid = image['file_name'].split('/')[1]
        frame_num = int(image['file_name'].split('.')[0].split('/')[-1])
        if frame_len_dict[vid] < 30:
            continue
        if frame_num > 20 or frame_num < 10:
            continue
        if f'{action}/{vid},{frame_num:05}' not in proposals.keys():
            continue
        if action_cnt[action] < action_thr:
            images.append(image)
            image_ids.append(image['id'])
            action_cnt[action] += 1
    print(action_cnt)

    annotations = []
    for annot in _annotations:
        if annot['image_id'] in image_ids:
            annotations.append(annot)

    info['images'] = images
    info['annotations'] = annotations
    with open(new_json_file, 'w') as njf:
        json.dump(info, njf, indent=4)

    return f'{name}_{action_thr * len(action_label)}'


if __name__ == '__main__':
    #get_pbtxt()
    #validate_csv()
    #print_frame_len()
    annot2json()
    name = sample_from_json('train')
    json2csv(name)
    name = sample_from_json('valid')
    json2csv(name)
