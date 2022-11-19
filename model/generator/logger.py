import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):

    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger
