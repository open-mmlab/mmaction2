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

def get_confusion_matrix(predictions, labels):
    confusion_mat = confusion_matrix(labels, predictions) 
    



def get_best_model_ckpt_path(run_dir):
    all_ckpts = glob(join(run_dir, '*.pth')) 
    for ckpt_path in all_ckpts:
        ckpt_name = ckpt_path.split('/')[-1]
        if 'best_' in ckpt_name:
            return ckpt_path 
    
    return None 
            

def get_train_test_domain_from_name(name):
    name = name.split('_')
    train = [] 
    test = []
    current = None 
    for part in name:
        if 'D' in part:
            if current == 'train':
                train.append(part) 
            else:
                test.append(part) 
        else:
            current = part 
    
    return train, test 


    
if __name__ == '__main__':


    # Self Supervised Evaluation Pickles 
    # self_supervised_runs = '/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/self-supervised-logits/'

    
    # domains = ['D1', 'D2', 'D3'] 
    # for test_domain in domains:
    #     run_dirs = glob(join(self_supervised_runs, f'*test_{test_domain}*')) 
    #     run_dirs = [r for r in run_dirs if 'ssl' not in r and 'clip' not in r]
    #     for run_dir in run_dirs:
    #         best_ckpt_path = get_best_model_ckpt_path(run_dir) 
    #         cmd = f"python tools/test.py /home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_selfsupervised_ek_rgb.py  {best_ckpt_path} --out {run_dir}/output.pkl --cfg-options data.test.domain=\'{test_domain}\'" 
    #         print(cmd)
    #         os.system(cmd)



    # TSM Baseline Evaluation Pickles
    tsm_baseline_runs = '/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline'
    tsm_baseline_format_single = 'train_[{}]_test_[{}]'
    tsm_baseline_format_multi = 'train_{}{}_test_{}' 
    domains = ['D1', 'D2', 'D3'] 
    multi_domain_dict = {1: ['2', '3'], 2: ['1', '3'], 3: ['1', '2']}
    best_epoch_ckpt_num = {(1, 1): 45, 
                            (1, 2): 20, 
                            (1, 3): 30, 
                            (2, 1): 10, 
                            (2, 2): 35, 
                            (2, 3): 25, 
                            (3, 1): 15, 
                            (3, 2): 10, 
                            (3, 3): 25} 
    for test_domain in domains:
        # single source domain 
        for train_domain in domains:
            train_domain_num = int(train_domain.replace('D', ''))
            test_domain_num = int(test_domain.replace('D', ''))

            tsm_work_dir = tsm_baseline_format_single.format(train_domain, test_domain) 
            tsm_work_dir = join(tsm_baseline_runs, tsm_work_dir) 
            best_ckpt_path = join(tsm_work_dir, f"best_top1_acc_epoch_{best_epoch_ckpt_num[(train_domain_num, test_domain_num)]}.pth")
            cmd = f"python tools/test.py /home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_rgb.py  {best_ckpt_path} --out {tsm_work_dir}/output.pkl --cfg-options data.test.domain=\'{test_domain}\'" 

            print(cmd)
            os.system(cmd)

        # multisource domains 

        domain_num = int(test_domain.replace('D', ''))
        train_domains = multi_domain_dict[domain_num]  
        tsm_work_dir = tsm_baseline_format_multi.format(f'd{train_domains[0]}', f'd{train_domains[1]}', test_domain.lower()) 
        tsm_work_dir = join(tsm_baseline_runs, tsm_work_dir) 
        best_ckpt_path = get_best_model_ckpt_path(tsm_work_dir) 
        cmd = f"python tools/test.py /home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_rgb.py  {best_ckpt_path} --out {tsm_work_dir}/output.pkl --cfg-options data.test.domain=\'{test_domain}\'"
        print(cmd)
        os.system(cmd)



    



    
    

