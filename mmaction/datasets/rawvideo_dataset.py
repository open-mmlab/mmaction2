import os.path as osp

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class RawVideoDataset(BaseDataset):
    """RawVideo dataset for action recognition, used in the Project OmniSource.

    The dataset loads clips of raw videos and apply specified transforms to
    return a dict containing the frame tensors and other information. Not that
    for this dataset, `multi_class` should be False.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath (without suffix), label, number of clips
    and index of positive clips (starting from 0), which are split with a
    whitespace. Raw videos should be first trimmed into 10 second clips,
    organized in the following format:

    .. code-block:: txt

        some/path/D32_1gwq35E/part_0.mp4
        some/path/D32_1gwq35E/part_1.mp4
        ......
        some/path/D32_1gwq35E/part_n.mp4

    Example of a annotation file:

    .. code-block:: txt

        some/path/D32_1gwq35E 66 10 0 1 2
        some/path/-G-5CJ0JkKY 254 5 3 4
        some/path/T4h1bvOd9DA 33 1 0
        some/path/4uZ27ivBl00 341 2 0 1
        some/path/0LfESFkfBSw 186 234 7 9 11
        some/path/-YIsNpBEx6c 169 100 9 10 11

    The first line indicates that the raw video `some/path/D32_1gwq35E` has
    action label `66`, consists of 10 clips (from `part_0.mp4` to
    `part_9.mp4`). The 1st, 2nd and 3rd clips are positive clips.


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        clipname_tmpl (str): The template of clip name in the raw video.
            Default: 'part_{}.mp4'.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 clipname_tmpl='part_{}.mp4',
                 **kwargs):
        super().__init__(ann_file, pipeline, start_index=0, **kwargs)
        assert self.multi_class is False
        self.clipname_tmpl = clipname_tmpl

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_dir = line_split[0]
                label = int(line_split[1])
                num_clips = int(line_split[2])
                positive_clip_inds = [int(ind) for ind in line_split[3:]]

                if self.data_prefix is not None:
                    video_dir = osp.join(self.data_prefix, video_dir)
                video_infos.append(
                    dict(
                        video_dir=video_dir,
                        label=label,
                        num_clips=num_clips,
                        positive_clip_inds=positive_clip_inds))
        return video_infos
