# ------------------------------------------------------------------------------
# Adapted from https://github.com/activitynet/ActivityNet/
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------
import argparse
import glob
import json
import os
import shutil
import ssl
import subprocess
import uuid
from collections import OrderedDict

import pandas as pd
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context


def create_video_folders(dataset, output_dir, tmp_dir):
    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = os.path.join(output_dir, 'test')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        # I should return a dict but ...
        return this_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    return label_to_dir


def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the output filename for a
    given video."""
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename


def download_clip(video_identifier,
                  output_filename,
                  start_time,
                  end_time,
                  tmp_dir='/tmp/kinetics/.tmp_dir',
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
        Indicates the beginning time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'

    status = False
    # Construct command line for getting the direct video link.
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
                except subprocess.CalledProcessError as err:
                    attempts += 1
                    if attempts == num_attempts:
                        return status, err.output
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
        except subprocess.CalledProcessError as err:
            return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    os.remove(tmp_filename)
    return status, 'Downloaded'


def download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir):
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(row, label_to_dir, trim_format)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists'])
        return status

    downloaded, log = download_clip(
        row['video-id'],
        output_filename,
        row['start-time'],
        row['end-time'],
        tmp_dir=tmp_dir)
    status = tuple([clip_id, downloaded, log])
    return status


def parse_kinetics_annotations(input_csv, ignore_is_cc=False):
    """Returns a parsed DataFrame.
    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'
    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = OrderedDict([('youtube_id', 'video-id'),
                               ('time_start', 'start-time'),
                               ('time_end', 'end-time'),
                               ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df


def main(input_csv,
         output_dir,
         trim_format='%06d',
         num_jobs=24,
         tmp_dir='/tmp/kinetics'):
    tmp_dir = os.path.join(tmp_dir, '.tmp_dir')

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)

    # Download all clips.
    if num_jobs == 1:
        status_list = []
        for _, row in dataset.iterrows():
            status_list.append(
                download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir))
    else:
        status_list = Parallel(
            n_jobs=num_jobs)(delayed(download_clip_wrapper)(
                row, label_to_dir, trim_format, tmp_dir)
                             for i, row in dataset.iterrows())

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)

    # Save download report.
    with open('download_report.json', 'w') as fobj:
        fobj.write(json.dumps(status_list))


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        'input_csv',
        type=str,
        help=('CSV file containing the following format: '
              'YouTube Identifier,Start time,End time,Class label'))
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
              'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    # help='CSV file of the previous version of Kinetics.')
    main(**vars(p.parse_args()))
