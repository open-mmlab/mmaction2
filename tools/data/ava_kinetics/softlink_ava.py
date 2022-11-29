# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument(
    '--ava_root',
    type=str,
    default='../../../data/ava',
    help='the path to save ava dataset')
p.add_argument(
    '--avakinetics_root',
    type=str,
    default='../../../data/ava_kinetics',
    help='the path to save ava-kinetics dataset')
args = p.parse_args()

ava_frames = os.path.abspath(args.ava_root + '/rawframes/')
kinetics_frames = os.path.abspath(args.avakinetics_root + '/rawframes/')

ava_folders = os.listdir(ava_frames)
for folder in ava_folders:
    cmd = 'ln -s %s/%s %s/%s' % (ava_frames, folder, kinetics_frames, folder)
    os.system(cmd)
