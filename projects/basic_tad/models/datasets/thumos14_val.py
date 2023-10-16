# adapted from basicTAD
import re
import warnings
from pathlib import Path

import mmengine
import numpy as np

from mmaction.datasets import BaseActionDataset
from mmaction.registry import DATASETS


def make_regex_pattern(fixed_pattern):
    # Use regular expression to extract number of digits
    num_digits = re.search(r'\{:(\d+)\}', fixed_pattern).group(1)
    # Build the pattern string using the extracted number of digits
    pattern = fixed_pattern.replace('{:' + num_digits + '}', r'\d{' + num_digits + '}')
    return pattern


@DATASETS.register_module()
class Thumos14ValDataset(BaseActionDataset):
    """Thumos14 dataset for temporal action detection."""

    metainfo = dict(classes=('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                             'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
                             'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                             'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                             'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
                             'VolleyballSpiking'))

    def __init__(self, clip_len=96, frame_interval=10, overlap_ratio=0.25, filename_tmpl='img_{:05}.jpg', **kwargs):
        self.filename_tmpl = filename_tmpl
        assert 0 <= overlap_ratio < 1
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.overlap_ratio = overlap_ratio

        self.ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        self.stride = int(self.ori_clip_len * (1 - self.overlap_ratio))

        super(Thumos14ValDataset, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        data = mmengine.load(self.ann_file)
        for video_name, video_info in data['database'].items():
            # Segments information
            segments = []
            labels = []
            ignore_flags = []
            for ann in video_info['annotations']:
                label = ann['label']
                segment = ann['segment']

                if not self.test_mode:
                    segment[0] = min(video_info['duration'], max(0, segment[0]))
                    segment[1] = min(video_info['duration'], max(0, segment[1]))
                    if segment[0] >= segment[1]:
                        continue

                if label in self.metainfo['classes']:
                    ignore_flags.append(0)
                    labels.append(self.metainfo['classes'].index(label))
                else:
                    ignore_flags.append(1)
                    labels.append(-1)
                segments.append(segment)
            if not segments:
                segments = np.zeros((0, 2))
                labels = np.zeros((0,))
                ignore_flags = np.zeros((0,))
            else:
                segments = np.array(segments)
                labels = np.array(labels)
                ignore_flags = np.array(ignore_flags)

            # Meta information
            frame_dir = Path(self.data_prefix['imgs']).joinpath(video_name)
            if not frame_dir.exists():
                warnings.warn(f'{frame_dir} does not exist.')
                continue
            pattern = make_regex_pattern(self.filename_tmpl)
            imgfiles = [img for img in frame_dir.iterdir() if re.fullmatch(pattern, img.name)]
            num_imgs = len(imgfiles)

            total_frames = num_imgs
            offset = 0
            idx = 1
            while True:
                if offset < total_frames - 1:
                    clip = offset + np.arange(self.clip_len) * self.frame_interval
                    clip = clip[clip < total_frames]
                    fps = round(num_imgs / video_info['duration'])
                    data_info = dict(video_name=f'{video_name}',
                                     frame_dir=str(frame_dir),
                                     duration=float(video_info['duration']),
                                     total_frames=num_imgs,
                                     filename_tmpl=self.filename_tmpl,
                                     fps=fps,
                                     frame_inds=clip,
                                     offset_sec=offset / fps,
                                     tsize=len(clip),
                                     num_clips=1,
                                     clip_len=self.clip_len,
                                     frame_interval=self.frame_interval,
                                     tscale_factor=fps / self.frame_interval,
                                     segments=segments.astype(np.float32),
                                     labels=labels.astype(np.int64),
                                     ignore_flags=ignore_flags.astype(np.float32))
                    data_list.append(data_info)
                    offset += self.stride
                    idx += 1
                else:
                    break

            # standard_ann_file = dict()
            # standard_ann_file['metainfo'] = dict(classes=self.CLASSES)
            # standard_ann_file['data_list'] = data_list
            # mmengine.dump(standard_ann_file, 'train.json')
        return data_list
