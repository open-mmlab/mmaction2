# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
from numpy.testing import assert_array_almost_equal

from mmaction.datasets import HVUDataset
from .base import BaseTestDataset


class TestHVUDataset(BaseTestDataset):

    def test_hvu_dataset(self):
        hvu_frame_dataset = HVUDataset(
            ann_file=self.hvu_frame_ann_file,
            pipeline=self.frame_pipeline,
            tag_categories=self.hvu_categories,
            tag_category_nums=self.hvu_category_nums,
            filename_tmpl=self.filename_tmpl,
            data_prefix=self.data_prefix,
            start_index=1)
        hvu_frame_infos = hvu_frame_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'imgs')
        assert hvu_frame_infos == [
            dict(
                frame_dir=frame_dir,
                total_frames=5,
                label=dict(
                    concept=[250, 131, 42, 51, 57, 155, 122],
                    object=[1570, 508],
                    event=[16],
                    action=[180],
                    scene=[206]),
                categories=self.hvu_categories,
                category_nums=self.hvu_category_nums,
                filename_tmpl=self.filename_tmpl,
                start_index=1,
                modality='RGB')
        ] * 2

        hvu_video_dataset = HVUDataset(
            ann_file=self.hvu_video_ann_file,
            pipeline=self.video_pipeline,
            tag_categories=self.hvu_categories,
            tag_category_nums=self.hvu_category_nums,
            data_prefix=self.data_prefix)
        hvu_video_infos = hvu_video_dataset.video_infos
        filename = osp.join(self.data_prefix, 'tmp.mp4')
        assert hvu_video_infos == [
            dict(
                filename=filename,
                label=dict(
                    concept=[250, 131, 42, 51, 57, 155, 122],
                    object=[1570, 508],
                    event=[16],
                    action=[180],
                    scene=[206]),
                categories=self.hvu_categories,
                category_nums=self.hvu_category_nums)
        ] * 2

        hvu_video_eval_dataset = HVUDataset(
            ann_file=self.hvu_video_eval_ann_file,
            pipeline=self.video_pipeline,
            tag_categories=self.hvu_categories_for_eval,
            tag_category_nums=self.hvu_category_nums_for_eval,
            data_prefix=self.data_prefix)

        results = [
            np.array([
                -1.59812844, 0.24459082, 1.38486497, 0.28801252, 1.09813449,
                -0.28696971, 0.0637848, 0.22877678, -1.82406999
            ]),
            np.array([
                0.87904563, 1.64264224, 0.46382051, 0.72865088, -2.13712525,
                1.28571358, 1.01320328, 0.59292737, -0.05502892
            ])
        ]
        mAP = hvu_video_eval_dataset.evaluate(results)
        assert_array_almost_equal(mAP['action_mAP'], 1.0)
        assert_array_almost_equal(mAP['scene_mAP'], 0.5)
        assert_array_almost_equal(mAP['object_mAP'], 0.75)
