# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import os
import platform
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove some unnecessary keys for smaller file size
    unnecessary_keys = ['message_hub', 'optimizer', 'param_schedulers']
    for k in unnecessary_keys:
        if k in checkpoint:
            del checkpoint[k]
    unnecessary_params = ['data_preprocessor.mean', 'data_preprocessor.std']
    for k in unnecessary_params:
        if k in checkpoint['state_dict']:
            del checkpoint['state_dict'][k]
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    if platform.system() == 'Windows':
        sha = subprocess.check_output(
            ['certutil', '-hashfile', out_file, 'SHA256'])
        sha = str(sha).split('\\r\\n')[1]
    else:
        sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file

    current_date = datetime.datetime.now().strftime('%Y%m%d')
    final_file = out_file_name + f'_{current_date}-{sha[:8]}.pth'
    os.rename(out_file, final_file)


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
