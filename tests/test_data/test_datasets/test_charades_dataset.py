import copy

from mmcv.utils import assert_dict_has_keys

from mmaction.datasets import CharadesDataset
from .base import BaseTestDataset


class TestCharadesDaataset(BaseTestDataset):

    def test_charades_dataset(self):
        _charades_test_pipeline = copy.deepcopy(self.charades_test_pipeline)
        _charades_test_pipeline[0]['test_mode'] = False
        charades_dataset = CharadesDataset(
            self.charades_ann_file,
            _charades_test_pipeline,
            self.data_prefix,
            filename_tmpl_prefix='{}',
            filename_tmpl_suffix='_{:06d}.jpg')
        charades_infos = charades_dataset.video_infos
        assert charades_infos[0]['filename_tmpl'] == 'YSKX3_{:06d}.jpg'

        _charades_test_pipeline = copy.deepcopy(self.charades_test_pipeline)
        charades_dataset = CharadesDataset(
            self.charades_ann_file,
            _charades_test_pipeline,
            self.data_prefix,
            filename_tmpl_prefix='img',
            filename_tmpl_suffix='_{:05d}.jpg',
            test_mode=True)
        charades_infos = charades_dataset.video_infos
        assert set(charades_infos[0]['label']) == set([75, 76, 77, 79])
        assert charades_infos[0]['filename_tmpl'] == 'img_{:05d}.jpg'

    def test_charades_pipeline(self):
        target_keys = [
            'frame_dir', 'total_frames', 'label', 'filename_tmpl',
            'start_index', 'modality'
        ]
        # CharadesDataset not in test mode
        _charades_test_pipeline = copy.deepcopy(self.charades_test_pipeline)
        _charades_test_pipeline[0]['test_mode'] = False
        charades_dataset = CharadesDataset(
            self.charades_ann_file,
            _charades_test_pipeline,
            self.data_prefix,
            filename_tmpl_prefix='img',
            filename_tmpl_suffix='_{:05d}.jpg')
        result = charades_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # CharadesDataset in test mode
        _charades_test_pipeline = copy.deepcopy(self.charades_test_pipeline)
        charades_dataset = CharadesDataset(
            self.charades_ann_file,
            _charades_test_pipeline,
            self.data_prefix,
            filename_tmpl_prefix='img',
            filename_tmpl_suffix='_{:05d}.jpg',
            test_mode=True)
        result = charades_dataset[0]
        assert assert_dict_has_keys(result, target_keys)
