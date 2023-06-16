# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
from subprocess import check_output

import mmengine


def get_duration(vid_name):
    command = f'ffprobe -i {vid_name} 2>&1 | grep "Duration"'
    output = str(check_output(command, shell=True))
    output = output.split(',')[0].split('Duration:')[1].strip()
    h, m, s = output.split(':')
    duration = int(h) * 3600 + int(m) * 60 + float(s)
    return duration


def trim(vid_name):
    try:
        lt = get_duration(vid_name)
    except Exception:
        print(f'get_duration failed for video {vid_name}', flush=True)
        return

    i = 0
    name, _ = osp.splitext(vid_name)

    # We output 10-second clips into the folder `name`
    dest = name
    mmengine.mkdir_or_exist(dest)

    command_tmpl = ('ffmpeg -y loglevel error -i {} -ss {} -t {} -crf 18 '
                    '-c:v libx264 {}/part_{}.mp4')
    while i * 10 < lt:
        os.system(command_tmpl.format(vid_name, i * 10, 10, dest, i))
        i += 1

    # remove a raw video after decomposing it into 10-second clip to save space
    os.remove(vid_name)


if __name__ == '__main__':
    vid_name = sys.argv[1]
    trim(vid_name)
