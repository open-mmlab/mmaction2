# This scripts is copied from
# https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/download.py  # noqa: E501
# The code is licensed under the MIT licence.
import os
import ssl
import subprocess

import mmcv
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context
data_file = '../../../data/ActivityNet'
video_list = f'{data_file}/video_info_new.csv'
anno_file = f'{data_file}/anet_anno_action.json'
output_dir = f'{data_file}/videos'


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
            except subprocess.CalledProcessError:
                attempts += 1
                if attempts == num_attempts:
                    return status, 'Fail'
            else:
                break
    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def download_clip_wrapper(youtube_id, output_dir):
    """Wrapper for parallel processing purposes."""
    # we do this to align with names in annotations
    output_filename = os.path.join(output_dir, 'v_' + youtube_id + '.mp4')
    if os.path.exists(output_filename):
        status = tuple(['v_' + youtube_id, True, 'Exists'])
        return status

    downloaded, log = download_clip(youtube_id, output_filename)
    status = tuple(['v_' + youtube_id, downloaded, log])
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
    # YoutubeIDs do not have prefix `v_`
    youtube_ids = [x.split(',')[0][2:] for x in lines]
    return youtube_ids


def main(input_csv, output_dir, anno_file, num_jobs=24):
    # Reading and parsing ActivityNet.
    youtube_ids = parse_activitynet_annotations(input_csv)

    # Creates folders where videos will be saved later.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Download all clips.
    if num_jobs == 1:
        status_list = []
        for index in youtube_ids:
            status_list.append(download_clip_wrapper(index, output_dir))
    else:
        status_list = Parallel(n_jobs=num_jobs)(
            delayed(download_clip_wrapper)(index, output_dir)
            for index in youtube_ids)

    # Save download report.
    mmcv.dump(status_list, 'download_report.json')
    annotation = mmcv.load(anno_file)
    downloaded = {status[0]: status[1] for status in status_list}
    annotation = {k: v for k, v in annotation.items() if downloaded[k]}
    anno_file_bak = anno_file.replace('.json', '_bak.json')
    os.system(f'mv {anno_file} {anno_file_bak}')
    mmcv.dump(annotation, anno_file)


if __name__ == '__main__':
    main(video_list, output_dir, anno_file, 24)
