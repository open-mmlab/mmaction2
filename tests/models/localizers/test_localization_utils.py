# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.models.localizers.utils import (generate_bsp_feature,
                                              generate_candidate_proposals,
                                              soft_nms, temporal_iop,
                                              temporal_iou)


def test_temporal_iou():
    anchors_min = np.array([0.0, 0.5])
    anchors_max = np.array([1.0, 1.5])
    box_min = 0.5
    box_max = 1.0

    iou = temporal_iou(anchors_min, anchors_max, box_min, box_max)
    assert_array_equal(iou, np.array([0.5, 0.5]))


def test_temporal_iop():
    anchors_min = np.array([0.0, 0.5])
    anchors_max = np.array([1.0, 1.5])
    box_min = 0.4
    box_max = 1.1

    ioa = temporal_iop(anchors_min, anchors_max, box_min, box_max)
    assert_array_almost_equal(ioa, np.array([0.6, 0.6]))


def test_soft_nms():
    proposals = np.array([[0., 1., 1., 1., 0.5, 0.5],
                          [0., 0.4, 1., 1., 0.4, 0.4],
                          [0., 0.95, 1., 1., 0.6, 0.6]])
    proposal_list = soft_nms(proposals, 0.75, 0.65, 0.9, 1)
    assert_array_equal(proposal_list, [[0., 0.95, 0.6], [0., 0.4, 0.4]])


def test_generate_candidate_proposals():
    video_list = [0, 1]
    video_infos = [
        dict(
            video_name='v_test1',
            duration_second=100,
            duration_frame=1000,
            annotations=[{
                'segment': [30.0, 60.0],
                'label': 'Rock climbing'
            }],
            feature_frame=900),
        dict(
            video_name='v_test2',
            duration_second=100,
            duration_frame=1000,
            annotations=[{
                'segment': [6.0, 8.0],
                'label': 'Drinking beer'
            }],
            feature_frame=900)
    ]
    tem_results_dir = osp.normpath(
        osp.join(osp.dirname(__file__), '../../data/tem_results'))
    # test when tem_result_ext is not valid
    with pytest.raises(NotImplementedError):
        result_dict = generate_candidate_proposals(
            video_list,
            video_infos,
            tem_results_dir,
            5,
            0.5,
            tem_results_ext='unsupport_ext')
    # test without result_dict
    assert_result1 = np.array([
        [0.1, 0.7, 0.58390868, 0.35708317, 0.20850396, 0.55555556, 0.55555556],
        [0.1, 0.5, 0.58390868, 0.32605207, 0.19038463, 0.29411765, 0.41666667],
        [0.1, 0.3, 0.58390868, 0.26221931, 0.15311213, 0., 0.],
        [0.3, 0.7, 0.30626667, 0.35708317, 0.10936267, 0.83333333, 0.83333333],
        [0.3, 0.5, 0.30626667, 0.32605207, 0.09985888, 0.45454545, 0.83333333]
    ])
    assert_result2 = np.array(
        [[0.1, 0.3, 0.78390867, 0.3622193, 0.28394685, 0., 0.],
         [0.1, 0.7, 0.78390867, 0.35708317, 0.27992059, 0., 0.],
         [0.1, 0.5, 0.78390867, 0.32605207, 0.25559504, 0., 0.]])
    result_dict = generate_candidate_proposals(video_list, video_infos,
                                               tem_results_dir, 5, 0.5)

    assert_array_almost_equal(result_dict['v_test1'], assert_result1)
    assert_array_almost_equal(result_dict['v_test2'], assert_result2)

    # test with result_dict
    result_dict = {}
    generate_candidate_proposals(
        video_list,
        video_infos,
        tem_results_dir,
        5,
        0.5,
        result_dict=result_dict)

    assert_array_almost_equal(result_dict['v_test1'], assert_result1)
    assert_array_almost_equal(result_dict['v_test2'], assert_result2)


def test_generate_bsp_feature():
    video_list = [0, 1]
    video_infos = [
        dict(
            video_name='v_test1',
            duration_second=100,
            duration_frame=1000,
            annotations=[{
                'segment': [30.0, 60.0],
                'label': 'Rock climbing'
            }],
            feature_frame=900),
        dict(
            video_name='v_test2',
            duration_second=100,
            duration_frame=1000,
            annotations=[{
                'segment': [6.0, 8.0],
                'label': 'Drinking beer'
            }],
            feature_frame=900)
    ]
    tem_results_dir = osp.normpath(
        osp.join(osp.dirname(__file__), '../../data/tem_results'))
    pgm_proposals_dir = osp.normpath(
        osp.join(osp.dirname(__file__), '../../data/proposals'))

    # test when extension is not valid
    with pytest.raises(NotImplementedError):
        result_dict = generate_bsp_feature(
            video_list,
            video_infos,
            tem_results_dir,
            pgm_proposals_dir,
            tem_results_ext='unsupport_ext')

    with pytest.raises(NotImplementedError):
        result_dict = generate_bsp_feature(
            video_list,
            video_infos,
            tem_results_dir,
            pgm_proposals_dir,
            pgm_proposal_ext='unsupport_ext')

    # test without result_dict
    result_dict = generate_bsp_feature(
        video_list, video_infos, tem_results_dir, pgm_proposals_dir, top_k=2)
    assert_result1 = np.array(
        [[
            0.02633105, 0.02489364, 0.02345622, 0.0220188, 0.02058138,
            0.01914396, 0.01770654, 0.01626912, 0.01541432, 0.01514214,
            0.01486995, 0.01459776, 0.01432558, 0.01405339, 0.01378121,
            0.01350902, 0.03064331, 0.02941124, 0.02817916, 0.02694709,
            0.02571502, 0.02448295, 0.02325087, 0.0220188, 0.01432558,
            0.01409228, 0.01385897, 0.01362567, 0.01339237, 0.01315907,
            0.01292577, 0.01269246
        ],
         [
             0.01350902, 0.01323684, 0.01296465, 0.01269246, 0.01242028,
             0.01214809, 0.01187591, 0.01160372, 0.01154264, 0.01169266,
             0.01184269, 0.01199271, 0.01214273, 0.01229275, 0.01244278,
             0.0125928, 0.01432558, 0.01409228, 0.01385897, 0.01362567,
             0.01339237, 0.01315907, 0.01292577, 0.01269246, 0.01214273,
             0.01227132, 0.01239991, 0.0125285, 0.0126571, 0.01278569,
             0.01291428, 0.01304287
         ]])
    assert_result2 = np.array(
        [[
            0.04133105, 0.03922697, 0.03712288, 0.0350188, 0.03291471,
            0.03081063, 0.02870654, 0.02660246, 0.02541432, 0.02514214,
            0.02486995, 0.02459776, 0.02432558, 0.02405339, 0.02378121,
            0.02350902, 0.04764331, 0.04583981, 0.04403631, 0.04223281,
            0.0404293, 0.0386258, 0.0368223, 0.0350188, 0.02432558, 0.02409228,
            0.02385897, 0.02362567, 0.02339237, 0.02315907, 0.02292577,
            0.02269246
        ],
         [
             0.02350902, 0.02323684, 0.02296465, 0.02269246, 0.02242028,
             0.02214809, 0.02187591, 0.02160372, 0.02120931, 0.02069266,
             0.02017602, 0.01965937, 0.01914273, 0.01862609, 0.01810944,
             0.0175928, 0.02432558, 0.02409228, 0.02385897, 0.02362567,
             0.02339237, 0.02315907, 0.02292577, 0.02269246, 0.01914273,
             0.01869989, 0.01825706, 0.01781422, 0.01737138, 0.01692854,
             0.0164857, 0.01604287
         ]])
    assert_array_almost_equal(result_dict['v_test1'], assert_result1)
    assert_array_almost_equal(result_dict['v_test2'], assert_result2)

    # test with result_dict
    result_dict = {}
    generate_bsp_feature(
        video_list,
        video_infos,
        tem_results_dir,
        pgm_proposals_dir,
        top_k=2,
        result_dict=result_dict)
    assert_array_almost_equal(result_dict['v_test1'], assert_result1)
    assert_array_almost_equal(result_dict['v_test2'], assert_result2)
