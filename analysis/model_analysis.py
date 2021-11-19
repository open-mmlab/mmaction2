# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from collections import Counter 
import numpy as np 
from glob import glob 
from os.path import join 
import os 
import pickle 
import pandas as pd 
from collections import Counter 
import cv2 
from tqdm import tqdm
import imageio
import wandb
from demo import demo_gradcam
from sys import exit 



def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        g = pickle.load(f) 
    return g 



def write_gif(frames, write_path):
    with imageio.get_writer(write_path, mode='I') as writer:
        for image in frames:
            # image = imageio.imread(filename)
            writer.append_data(image)



def generate_html(write_dir):
    gifs = glob(join(write_dir, '*.gif'))
    html_code = f"""
    <html>
    <head>
    <title>
    {write_dir}
    </title>
    </head>
    <ol>
    """
    for gif in gifs:
        html_code += f"""
        <li><img src={gif.split('/')[-1]}></li>\n\n
        """
    
    html_code += """
    </html>
    """

    with open(f'{write_dir}/index.html', 'w') as  f:
        f.write(html_code) 

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Analysis of a single model') 
    

    # m1 should be the model whose predictions need to be compared with a baseline m2. 
    parser.add_argument('-m1', '--model-1-pkl', required=True, type=str) 
    parser.add_argument('-t', '--test-domain', required=True, type=int)
    parser.add_argument('-w', '--write-dir', default=None, type=str, help='Output Location of the video')
    parser.add_argument('-v', '--verbose', action='store_true', help='Toggle to print individual statistics')
    parser.add_argument('-g', '--generate-gifs', action='store_true', help='toggle to generate gifs')
    parser.add_argument('-wd', '--wandb', default=None, help='Description of the run')
    parser.add_argument('-gradcam', '--generate-gradcam', action='store_true', help='Toggle generation of gradcams')
    parser.add_argument('--config-name', type=str, default=None, help='Config name -- needed only if gradcam is run')
    parser.add_argument('--ckpt-path', type=str, default=None, help='Checkpoint path -- needed only if gradcam is run')

    args = parser.parse_args() 
    return args 

def get_verbclass_to_verb_mapping(data_labels):
    mapping = {}
    for data in data_labels:
        mapping[data['verb_class']] = data['verb'] 
    return mapping 

 
def load_annotations(domain_pickle_file_path):
    df = pd.read_pickle(domain_pickle_file_path) 
    data_labels = []
    count = 0
    for _, line in df.iterrows():
        participant_id = line['participant_id'] 
        video_id = line['video_id']                 
        start_frame = int(line['start_frame']) 
        end_frame = int(line['stop_frame'])
        verb = line['verb']
        label = line['verb_class'] 
        label = int(label)
        metadata_dict = {"index": count, "video_id": video_id, "verb_class": label, "verb": verb, **line}
        data_labels.append(metadata_dict)
        count += 1
    
    return data_labels

def get_preds(model_pkl):
    preds = [] 
    for pred_logits in model_pkl:
        preds.append(np.argmax(pred_logits)) 
    
    return preds 

def get_correct_or_wrong_predictions(test_labels, model_pkl, correct=True):
    result_preds = [] 
    for metadata, pred_logits in zip(test_labels, model_pkl):
        pred = np.argmax(pred_logits) 
        gt = metadata['verb_class'] 
        if correct and pred == int(gt):
            result_preds.append(metadata["index"]) 
        elif not correct and pred != int(gt):
            result_preds.append(metadata["index"])
    
    return result_preds

def put_text(image, text, is_pred=False):
    loc = (0, 10)
    height, width, layer = image.shape 
    if is_pred:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255) 
    

    if is_pred:
        loc =  (0, height - 10) 
    else:
        loc = (0, 20) 

    text = f'pred: {text}' if is_pred else f'gt: {text}' 
    fontScale = 1
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, text, loc, font, fontScale, color, thickness, cv2.LINE_AA)
    return image 

def join_frames_to_video(data, domain, write_dir=None, pred_cls=None, gt_cls=None, args=None):

    start_frame = data['start_frame']
    stop_frame = data['stop_frame'] 
    video_id = data['video_id'] 

    frame_fmt = 'frame_{:010d}.jpg' 
    frame_path_fmt = f'/home/ubuntu/datasets/action_recognition/EPIC_KITCHENS_UDA/frames_rgb_flow'\
                 f'/rgb/test/D{domain}/{video_id}/{frame_fmt}'
    
    images = [] 

    for i in range(start_frame, stop_frame + 1):
        frame_path = frame_path_fmt.format(i) 
        image = cv2.imread(frame_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, layer = image.shape

        if pred_cls:
            image = put_text(image, pred_cls, is_pred=True) 
        if gt_cls:
            image = put_text(image, gt_cls, is_pred=False)

        size = (width, height) 
        images.append(image) 

    os.makedirs(write_dir, exist_ok=True)
    write_path = f'{write_dir}/video_{video_id}_{start_frame}.gif'

    write_gif(images, write_path)


def get_grad_cam(data, domain, config_name, ckpt_path, write_dir=None):
    start_frame = data['start_frame']
    stop_frame = data['stop_frame'] 
    video_id = data['video_id'] 

    frame_fmt = 'frame_{:010d}.jpg' 
    frame_path_fmt = f'/home/ubuntu/datasets/action_recognition/EPIC_KITCHENS_UDA/frames_rgb_flow'\
                 f'/rgb/test/D{domain}/{video_id}/{frame_fmt}'
    video_folder = f'/home/ubuntu/datasets/action_recognition/EPIC_KITCHENS_UDA/frames_rgb_flow'\
                 f'/rgb/test/D{domain}/{video_id}'
    
    os.makedirs(write_dir, exist_ok=True)
    grad_cam_cmd = f"python demo/demo_gradcam.py {config_name}   {ckpt_path} {video_folder} --cfg-options model.backbone.frozen_stages=-1 data.test.start_index={start_frame} dataset_type='RawframeDataset' data.test.end_frame={stop_frame} --use-frames --out-filename {write_dir}/gradcam_{video_id}_{start_frame}.gif" 
    os.system(grad_cam_cmd)

    

if __name__ == '__main__':
    args = parse_args() 
    test_domain_pickle_file_path = f"/home/ubuntu/users/maiti/projects/MM-SADA_Domain_Adaptation_Splits/D{args.test_domain}_test.pkl"
    test_labels = load_annotations(test_domain_pickle_file_path) 
    verbclass_2_verb_mapping = get_verbclass_to_verb_mapping(test_labels)

    model_1_pkl_path = args.model_1_pkl 
    model_1_pkl = read_pickle(model_1_pkl_path) 

    model_1_wrong_preds = get_correct_or_wrong_predictions(test_labels, model_1_pkl, correct=False) 

    # Correct by 1st and wrong by the 2nd model

    missclassified_classes = [] 
    video_ids = [] 

    if args.verbose:
        for idx, index in enumerate(model_1_wrong_preds):
            print(f'S. No: {idx+1}')
            print(test_labels[index]) 
            missclassified_classes.append(test_labels[index]['verb']) 
            video_ids.append(test_labels[index]['video_id'])
        
        print('##############################################################################')
        print('Summary Of Misclassified Verbs')
        print(Counter(missclassified_classes))
        print('Summary of Missclassified Videos') 
        print(Counter(video_ids))
        print('##############################################################################')

    if args.generate_gifs:
        print('Generating Videos of the mistakes...')
        for index in tqdm(model_1_wrong_preds):
            data = test_labels[index] 
            pred_cls = verbclass_2_verb_mapping[np.argmax(model_1_pkl[index])]
            gt_cls = data['verb'] 
            join_frames_to_video(data, args.test_domain, write_dir=args.write_dir, pred_cls=pred_cls, gt_cls=gt_cls, args=args)
    
    if args.generate_gradcam:
        assert args.ckpt_path, 'Please give checkpoint path for gradcam' 
        assert args.config_name, 'Please give config path for gradcam'
        print('Generating gradcam videos')
        for index in tqdm(model_1_wrong_preds):
            data = test_labels[index] 
            get_grad_cam(data, args.test_domain, args.config_name, args.ckpt_path, write_dir=args.write_dir) 

    if args.wandb:
        wandb.init(project='action-recognition', 
                   notes=args.wandb.split('_')[0],
                   name=args.wandb.split('_')[1] 
                   )
        video_gifs = glob(join(args.write_dir, 'video_*.gif')) 
        gradcam_gifs = glob(join(args.write_dir, 'gradcam_*.gif'))
        for video_gif, gradcam_gif in zip(video_gifs, gradcam_gifs):
            wandb.log({"Failure Cases": wandb.Image(video_gif, caption=video_gif.split('/')[-1]), 
                    "GradCam": wandb.Image(gradcam_gif, caption=gradcam_gif.split('/')[-1])})

        all_preds = get_preds(model_1_pkl)
        all_gt = [k['verb_class'] for k in test_labels]
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=all_gt, preds=all_preds,
                        class_names=[verbclass_2_verb_mapping[k] for k in range(8)])}) 


        
        
        