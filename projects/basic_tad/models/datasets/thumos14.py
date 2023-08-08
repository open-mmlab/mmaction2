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
class Thumos14Dataset(BaseActionDataset):
    """Thumos14 dataset for temporal action detection."""

    metainfo = dict(classes=('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                              'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
                              'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                              'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                              'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
                              'VolleyballSpiking'))

    def __init__(self, filename_tmpl='img_{:05}.jpg', **kwargs):
        self.filename_tmpl = filename_tmpl
        super(Thumos14Dataset, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        data = mmengine.load(self.ann_file)
        for video_name, video_info in data['database'].items():
            # Meta information
            frame_dir = Path(self.data_prefix['imgs']).joinpath(video_name)
            pattern = make_regex_pattern(self.filename_tmpl)
            imgfiles = [img for img in frame_dir.iterdir() if re.fullmatch(pattern, img.name)]
            num_imgs = len(imgfiles)

            data_info = dict(video_name=video_name,
                             frame_dir=str(frame_dir),
                             duration=float(video_info['duration']),
                             total_frames=num_imgs,
                             filename_tmpl=self.filename_tmpl,
                             fps=int(round(num_imgs / video_info['duration'])))

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

            if not segments or np.all(ignore_flags):
                warnings.warn(f'No valid segments found in video {video_name}. Excluded')
                continue

            data_info.update(dict(
                segments=np.array(segments, dtype=np.float32),
                labels=np.array(labels, dtype=np.int64),
                ignore_flags=np.array(ignore_flags, dtype=np.float32)))

            data_list.append(data_info)

            # standard_ann_file = dict()
            # standard_ann_file['metainfo'] = dict(classes=self.CLASSES)
            # standard_ann_file['data_list'] = data_list
            # mmengine.dump(standard_ann_file, 'train.json')
        return data_list
