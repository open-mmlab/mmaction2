# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.datasets import RepeatDataset
from .base import BaseTestDataset


class TestRepeatDataset(BaseTestDataset):

    def test_repeat_dataset(self):
        dataset_cfg = dict(
            type='RawframeDataset',
            ann_file=self.frame_ann_file,
            pipeline=self.frame_pipeline,
            data_prefix=self.data_prefix)

        repeat_dataset = RepeatDataset(dataset_cfg, 5)
        assert len(repeat_dataset) == 10
        result_a = repeat_dataset[0]
        result_b = repeat_dataset[2]
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
