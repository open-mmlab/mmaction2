# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger

from mmengine.logging.logger import MMLogger


def get_root_logger():
    return MMLogger.get_current_instance()
