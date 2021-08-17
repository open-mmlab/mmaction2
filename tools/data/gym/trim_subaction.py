# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess

import mmcv

data_root = '../../../data/gym'
anno_root = f'{data_root}/annotations'

event_anno_file = f'{anno_root}/event_annotation.json'
event_root = f'{data_root}/events'
subaction_root = f'{data_root}/subactions'

events = os.listdir(event_root)
events = set(events)
annotation = mmcv.load(event_anno_file)

mmcv.mkdir_or_exist(subaction_root)

for k, v in annotation.items():
    if k + '.mp4' not in events:
        print(f'video {k[:11]} has not been downloaded '
              f'or the event clip {k} not generated')
        continue

    video_path = osp.join(event_root, k + '.mp4')

    for subaction_id, subaction_anno in v.items():
        timestamps = subaction_anno['timestamps']
        start_time, end_time = timestamps[0][0], timestamps[-1][1]
        subaction_name = k + '_' + subaction_id

        output_filename = subaction_name + '.mp4'

        command = [
            'ffmpeg', '-i',
            '"%s"' % video_path, '-ss',
            str(start_time), '-t',
            str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'copy',
            '-threads', '8', '-loglevel', 'panic',
            '"%s"' % osp.join(subaction_root, output_filename)
        ]
        command = ' '.join(command)
        try:
            subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(
                f'Trimming of the Subaction {subaction_name} of Event '
                f'{k} Failed',
                flush=True)
