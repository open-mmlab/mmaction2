# This scripts is copied from
# https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py  # noqa: E501
import argparse
import json
import os
import subprocess

import ssl  # isort:skip

from joblib import Parallel, delayed  # isort:skip

ssl._create_default_https_context = ssl._create_unverified_context


def download_clip(video_identifier,
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
            except subprocess.CalledProcessError as err:
                attempts += 1
                if attempts == num_attempts:
                    return status, err.output
            else:
                break
    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def download_clip_wrapper(youtube_id, output_dir):
    """Wrapper for parallel processing purposes."""
    output_filename = os.path.join(output_dir, youtube_id + '.mp4')
    if os.path.exists(output_filename):
        status = tuple([youtube_id, True, 'Exists'])
        return status

    downloaded, log = download_clip(youtube_id, output_filename)
    status = tuple([youtube_id, downloaded, log])
    return status


def parse_activitynet_annotations(input_csv):
    """Returns a list of YoutubeID.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'video,numFrame,seconds,fps,rfps,subset,featureFrame'
    returns:
    -------
    youtube_ids: list
        List of all YoutubeIDs in ActivityNet.

    """
    lines = open(input_csv).readlines()
    lines = lines[1:]
    youtube_ids = [x.split(',')[0] for x in lines]
    return youtube_ids


def main(input_csv, output_dir, num_jobs=24):
    # Reading and parsing ActivityNet.
    youtube_ids = parse_activitynet_annotations(input_csv)

    # Creates folders where videos will be saved later.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Download all clips.
    if num_jobs == 1:
        status_lst = []
        for index in youtube_ids:
            status_lst.append(download_clip_wrapper(index, output_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(
            delayed(download_clip_wrapper)(index, output_dir)
            for index in youtube_ids)

    # Save download report.
    with open('download_report.json', 'w') as fobj:
        fobj.write(json.dumps(status_lst))


if __name__ == '__main__':
    description = 'Helper script for downloading ActivityNet videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        'input_csv',
        type=str,
        help=('CSV file containing the following format: '
              'video,numFrame,seconds,fps,rfps,subset,featureFrame'
              'video follows the format: v_YoutubeID'))
    p.add_argument(
        'output_dir',
        type=str,
        help='Output directory where videos will be saved.')
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    # help='CSV file of the previous version of Kinetics.')
    main(**vars(p.parse_args()))
