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
from sklearn.metrics import confusion_matrix 
from collections import Counter 
import numpy as np 
from glob import glob 
from os.path import join 
import os 
import pickle 
import pandas as pd 
from collections import Counter 


def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        g = pickle.load(f) 
    return g 

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Outputs of two models') 
    
    parser.add_argument('--model-1', required=True, type=str) 
    parser.add_argument('--model-2', required=True, type=str)
    parser.add_argument('--test-domain', required=True, type=int)

    args = parser.parse_args() 
    return args 

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
        metadata_dict = {"index": count, "video_id": video_id, "verb_class": label, "verb": verb}
        data_labels.append(metadata_dict)
        count += 1
    
    return data_labels
    
def get_correct_predictions(test_labels, model_pkl):
    correct_preds = [] 
    for metadata, pred_logits in zip(test_labels, model_pkl):
        pred = np.argmax(pred_logits) 
        gt = metadata['verb_class'] 
        if pred == int(gt):
            correct_preds.append(metadata["index"]) 
    
    return correct_preds


if __name__ == '__main__':
    args = parse_args() 
    test_domain_pickle_file_path = f"/home/ubuntu/users/maiti/projects/MM-SADA_Domain_Adaptation_Splits/D{args.test_domain}_test.pkl"
    test_labels = load_annotations(test_domain_pickle_file_path) 
    model_1_pkl_path = join(args.model_1, 'output.pkl') 
    model_2_pkl_path = join(args.model_2, 'output.pkl') 
    model_1_pkl = read_pickle(model_1_pkl_path) 
    model_2_pkl = read_pickle(model_2_pkl_path) 

    model_1_correct_preds = get_correct_predictions(test_labels, model_1_pkl) 
    model_2_correct_preds = get_correct_predictions(test_labels, model_2_pkl) 

    # Correct by 1st and wrong by the 2nd model

    diff_in_preds = set(model_1_correct_preds) - set(model_2_correct_preds) 
    missclassified_classes = [] 
    video_ids = [] 
    for index in diff_in_preds:
        print(test_labels[index]) 
        missclassified_classes.append(test_labels[index]['verb']) 
        video_ids.append(test_labels[index]['video_id'])
    
    print('##############################################################################')
    print('Summary Of Misclassified Verbs')
    print(Counter(missclassified_classes))
    print('Summary of Missclassified Videos') 
    print(Counter(video_ids))
    print('##############################################################################')








