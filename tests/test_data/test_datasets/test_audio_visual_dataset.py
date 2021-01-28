import os.path as osp

from mmaction.datasets import AudioVisualDataset
from .base import BaseTestDataset


class TestAudioVisualDataset(BaseTestDataset):

    def test_audio_visual_dataset(self):
        test_dataset = AudioVisualDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            video_prefix=self.data_prefix,
            data_prefix=self.data_prefix)
        video_infos = test_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'imgs')
        audio_path = osp.join(self.data_prefix, 'imgs.npy')
        filename = osp.join(self.data_prefix, 'imgs.mp4')
        assert video_infos == [
            dict(
                frame_dir=frame_dir,
                audio_path=audio_path,
                filename=filename,
                total_frames=5,
                label=127)
        ] * 2
        assert test_dataset.start_index == 1
