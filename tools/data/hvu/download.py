# ------------------------------------------------------------------------------
# Adapted from https://github.com/activitynet/ActivityNet/
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import glob
import os
import shutil
import ssl
import subprocess
import uuid

import mmcv
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context
args = None


def create_video_folders(dataset, output_dir, tmp_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)


def construct_video_filename(item, trim_format, output_dir):
    """Given a dataset row, this function constructs the output filename for a
    given video."""
    youtube_id, start_time, end_time = item
    start_time, end_time = int(start_time * 10), int(end_time * 10)
    basename = '%s_%s_%s.mp4' % (youtube_id, trim_format % start_time,
                                 trim_format % end_time)
    output_filename = os.path.join(output_dir, basename)
    return output_filename


def download_clip(video_identifier,
                  output_filename,
                  start_time,
                  end_time,
                  tmp_dir='/tmp/hvu',
                  num_attempts=5,
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'

    status = False
    tmp_filename = os.path.join(tmp_dir, '%s.%%(ext)s' % uuid.uuid4())

    if not os.path.exists(output_filename):
        if not os.path.exists(tmp_filename):
            command = [
                'youtube-dl', '--quiet', '--no-warnings',
                '--no-check-certificate', '-f', 'mp4', '-o',
                '"%s"' % tmp_filename,
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
                        return status, 'Downloading Failed'
                else:
                    break

        tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
        # Construct command to trim the videos (ffmpeg required).
        command = [
            'ffmpeg', '-i',
            '"%s"' % tmp_filename, '-ss',
            str(start_time), '-t',
            str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'copy',
            '-threads', '1', '-loglevel', 'panic',
            '"%s"' % output_filename
        ]
        command = ' '.join(command)
        try:
            subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            return status, 'Trimming Failed'

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    os.remove(tmp_filename)
    return status, 'Downloaded'


def download_clip_wrapper(item, trim_format, tmp_dir, output_dir):
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(item, trim_format, output_dir)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists'])
        return status

    youtube_id, start_time, end_time = item
    downloaded, log = download_clip(
        youtube_id, output_filename, start_time, end_time, tmp_dir=tmp_dir)

    status = tuple([clip_id, downloaded, log])
    return status


def parse_hvu_annotations(input_csv):
    """Returns a parsed DataFrame.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'Tags, youtube_id, time_start, time_end'
    returns:
    -------
    dataset: List of tuples. Each tuple consists of
        (youtube_id, time_start, time_end). The type of time is float.
    """
    lines = open(input_csv).readlines()
    lines = [x.strip().split(',')[1:] for x in lines[1:]]

    lines = [(x[0], float(x[1]), float(x[2])) for x in lines]

    return lines


def main(input_csv,
         output_dir,
         trim_format='%06d',
         num_jobs=24,
         tmp_dir='/tmp/hvu'):
    # Reading and parsing HVU.
    dataset = parse_hvu_annotations(input_csv)

    # Creates folders where videos will be saved later.
    create_video_folders(dataset, output_dir, tmp_dir)

    # Download all clips.
    if num_jobs == 1:
        status_lst = []
        for item in dataset:
            status_lst.append(
                download_clip_wrapper(item, trim_format, tmp_dir, output_dir))
    else:
        status_lst = Parallel(n_jobs=num_jobs)(
            delayed(download_clip_wrapper)(item, trim_format, tmp_dir,
                                           output_dir) for item in dataset)

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)
    # Save download report.
    mmcv.dump(status_lst, 'download_report.json')


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming HVU videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        'input_csv',
        type=str,
        help=('CSV file containing the following format: '
              'Tags, youtube_id, time_start, time_end'))
    p.add_argument(
        'output_dir',
        type=str,
        help='Output directory where videos will be saved.')
    p.add_argument(
        '-f',
        '--trim-format',
        type=str,
        default='%06d',
        help=('This will be the format for the '
              'filename of trimmed videos: '
              'videoid_%0xd(start_time)_%0xd(end_time).mp4. '
              'Note that the start_time is multiplied by 10 since '
              'decimal exists somewhere. '))
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/hvu')
    main(**vars(p.parse_args()))
