import logging
from mmcv.runner import get_dist_info


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger
