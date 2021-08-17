# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.datasets import ConcatDataset
from .base import BaseTestDataset


class TestConcatDataset(BaseTestDataset):

    def test_concat_dataset(self):
        dataset_cfg = dict(
            type='RawframeDataset',
            ann_file=self.frame_ann_file,
            pipeline=self.frame_pipeline,
            data_prefix=self.data_prefix)
        repeat_dataset_cfg = dict(
            type='RepeatDataset', times=2, dataset=dataset_cfg)

        concat_dataset = ConcatDataset(
            datasets=[dataset_cfg, repeat_dataset_cfg])

        assert len(concat_dataset) == 6
        result_a = concat_dataset[0]
        result_b = concat_dataset[4]
        assert set(result_a) == set(result_b)
        for key in result_a:
            if isinstance(result_a[key], np.ndarray):
                assert np.equal(result_a[key], result_b[key]).all()
            elif isinstance(result_a[key], list):
                assert all(
                    np.array_equal(a, b)
                    for (a, b) in zip(result_a[key], result_b[key]))
            else:
                assert result_a[key] == result_b[key]
