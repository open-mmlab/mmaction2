import os.path as osp

from mmaction.datasets import RawVideoDataset
from .base import BaseTestDataset


class TestRawVideoDataset(BaseTestDataset):

    def test_rawvideo_dataset(self):
        # Try to load txt file
        rawvideo_dataset = RawVideoDataset(
            ann_file=self.rawvideo_test_anno_txt,
            pipeline=self.rawvideo_pipeline,
            clipname_tmpl='part_{}.mp4',
            sampling_strategy='positive',
            data_prefix=self.data_prefix)
        result = rawvideo_dataset[0]
        clipname = osp.join(self.data_prefix, 'rawvideo_dataset', 'part_0.mp4')
        assert result['filename'] == clipname

        # Try to load json file
        rawvideo_dataset = RawVideoDataset(
            ann_file=self.rawvideo_test_anno_json,
            pipeline=self.rawvideo_pipeline,
            clipname_tmpl='part_{}.mp4',
            sampling_strategy='random',
            data_prefix=self.data_prefix,
            test_mode=True)
        result = rawvideo_dataset[0]
