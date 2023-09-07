# Copyright (c) OpenMMLab. All rights reserved.
"""Used to compress videos (FPS and dimensions) in the Singularity project.

copied from https://github.com/klauscc/VindLU
"""
import argparse
import os
import shutil
import subprocess
from multiprocessing import Pool
from os.path import exists, join
from pathlib import Path

try:
    from psutil import cpu_count
except ImportError:
    from multiprocessing import cpu_count

from functools import partial

from PIL import Image
from tqdm import tqdm


def resize_image(input_path, output_path, size=224):
    with Image.open(input_path) as img:
        w, h = img.width, img.height
        r = 1. * w / h
        if w > h:
            h = size
            w = r * size
        else:
            h = size / r
            w = size

        img_resized = img.resize((int(w), int(h)))
        img_resized.save(output_path)


def _compress_images(input_output_pair, size=224):
    """Scale and downsample an input image to a given fps and size (shorter
    side size).

    This also removes the audio from the image.
    """
    input_image_path, output_image_path = input_output_pair
    try:
        resize_image(input_image_path, output_image_path, size)
    except Exception as e:
        print(f'Caught Exception {e}')


def _compress_videos(input_output_pair, size=224, fps=3):
    """Scale and downsample an input video to a given fps and size (shorter
    side size).

    This also removes the audio from the video.
    """
    input_file_path, output_file_path = input_output_pair
    try:
        command = [
            'ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-i',
            input_file_path,
            '-filter:v',  # no audio
            f"scale='if(gt(a,1),trunc(oh*a/2)*2,{size})':'if(gt(a,1),{size},trunc(ow*a/2)*2)'",  # noqa: E501
            '-map',
            '0:v',  # no audio
            '-r',
            str(fps),  # frames per second
            # '-g', str(16),
            output_file_path,
        ]
        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        raise e


def _compress(input_output_pair, fps=3, size=224, file_type='image'):
    if file_type == 'image':
        _compress_images(input_output_pair, size)
    elif file_type == 'video':
        _compress_videos(input_output_pair, size, fps)


def prepare_input_output_pairs(input_root,
                               output_root,
                               input_file_list_path=None):
    # filename list in `input_file_list_path` can be created very fast using `ls -U . >> ../video_filenames.txt`  # noqa: E501
    if input_file_list_path:
        with open(input_file_list_path, 'r') as f:
            filenames = [s.strip() for s in f.readlines()]
    else:
        filenames = [
            video_path.name for video_path in Path(input_root).glob('*.mp4')
        ]
    print(f'There are {len(filenames)} video/images files loaded from list.')
    input_file_path_list = []
    output_file_path_list = []
    for e in tqdm(filenames, desc='find un-processed videos/images'):
        input_file_path = join(input_root, e)
        output_file_path = join(output_root, e)
        if not exists(output_file_path):
            input_file_path_list.append(input_file_path)
            output_file_path_list.append(output_file_path)
    return input_file_path_list, output_file_path_list


def run_compress():
    parser = argparse.ArgumentParser(
        description='Compress videos/images for speed-up')
    parser.add_argument(
        '--input_root', type=str, help='input root', required=True)
    parser.add_argument(
        '--input_file_list_path',
        type=str,
        default=None,
        help='list of video filenames under args.input_root, it can be '
        'created efficiently with `ls -U /path/to/video >> /path/to/video_filenames.txt`'  # noqa: E501
    )
    parser.add_argument(
        '--output_root', type=str, help='output root', required=True)
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='shorter side size, aspect ratio is kept')
    parser.add_argument('--num_workers', type=int, default=24, help='#workers')
    parser.add_argument(
        '--fps',
        type=int,
        default=3,
        help='fps for output video, ignored if file_type == image')
    parser.add_argument(
        '--file_type',
        type=str,
        choices=['image', 'video'],
        help='input file type')
    args = parser.parse_args()

    # set paths
    input_root = args.input_root
    output_root = args.output_root
    assert input_root != output_root
    if not exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    # prepare and find un-processed
    input_file_path_list, output_file_path_list = prepare_input_output_pairs(
        input_root,
        output_root,
        input_file_list_path=args.input_file_list_path,
    )
    print(f'input_file_path_list[:3] {input_file_path_list[:3]}')
    print(f'output_file_path_list[:3] {output_file_path_list[:3]}')
    print('Total videos/images need to process: {}'.format(
        len(input_file_path_list)))

    # start parallel jobs
    num_cores = cpu_count()
    num_workers = args.num_workers
    print(
        f'Begin with {num_cores}-core logical processor, {num_workers} workers'
    )
    compress = partial(
        _compress, fps=args.fps, size=args.size, file_type=args.file_type)
    input_pairs = list(zip(input_file_path_list, output_file_path_list))
    with Pool(num_workers) as pool, tqdm(
            total=len(input_file_path_list),
            desc='re-encoding videos/images') as pbar:
        for idx, _ in enumerate(
                pool.imap_unordered(compress, input_pairs, chunksize=32)):
            pbar.update(1)

    # copy-paste failed files
    print('Compress finished, copy-paste failed files...')
    copy_count = 0
    for input_file_path, output_file_path in zip(input_file_path_list,
                                                 output_file_path_list):
        if exists(input_file_path):
            if exists(output_file_path) is False or os.path.getsize(
                    output_file_path) < 1.:
                copy_count += 1
                shutil.copyfile(input_file_path, output_file_path)
                print('Copy and replace file: {}'.format(output_file_path))
    print(f'copy_count {copy_count}')


if __name__ == '__main__':
    run_compress()
