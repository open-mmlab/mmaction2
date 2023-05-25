# Copyright (c) OpenMMLab. All rights reserved.
# This scripts is copied from
# https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py  # noqa: E501
# The code is licensed under the MIT licence.
import argparse
import os
import ssl
import subprocess

import mmengine
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context


def download(video_identifier,
             output_filename,
             num_attempts=5,
             url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'

    status = False

    if not os.path.exists(output_filename):
        command = [
            'youtube-dl', '--quiet', '--no-warnings', '--no-check-certificate',
            '-f', 'mp4', '-o',
            '"%s"' % output_filename,
            '"%s"' % (url_base + video_identifier)
        ]
        command = ' '.join(command)
        print(command)
        attempts = 0
        while True:
            try:
                subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                attempts += 1
                if attempts == num_attempts:
                    return status, 'Fail'
            else:
                break
    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def download_wrapper(youtube_id, output_dir):
    """Wrapper for parallel processing purposes."""
    # we do this to align with names in annotations
    output_filename = os.path.join(output_dir, youtube_id + '.mp4')
    if os.path.exists(output_filename):
        status = tuple([youtube_id, True, 'Exists'])
        return status

    downloaded, log = download(youtube_id, output_filename)
    status = tuple([youtube_id, downloaded, log])
    return status


def main(input, output_dir, num_jobs=24):
    # Reading and parsing ActivityNet.
    youtube_ids = mmengine.load(input).keys()
    # Creates folders where videos will be saved later.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Download all clips.
    if num_jobs == 1:
        status_list = []
        for index in youtube_ids:
            status_list.append(download_wrapper(index, output_dir))
    else:
        status_list = Parallel(n_jobs=num_jobs)(
            delayed(download_wrapper)(index, output_dir)
            for index in youtube_ids)

    # Save download report.
    mmengine.dump(status_list, 'download_report.json')


if __name__ == '__main__':
    description = 'Helper script for downloading GYM videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input', type=str, help='The gym annotation file')
    p.add_argument(
        'output_dir', type=str, help='Output directory to save videos.')
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    main(**vars(p.parse_args()))
