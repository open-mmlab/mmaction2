# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging.logger import MMLogger


def get_root_logger():
    """Get root logger.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """

    return MMLogger.get_current_instance()
