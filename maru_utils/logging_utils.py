# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

def init_logger(log_file=None, log_file_level=logging.INFO):                            # shoudl this be None?
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] [%(filename)s: %(lineno)d] %(message)s")          # set log message format
    logger = logging.getLogger(__name__)                                                # why no name arg?
    logger.setLevel(log_file_level)

    console_handler = logging.StreamHandler()                        # new instance of StreamHandler (to print to console)
    console_handler.setFormatter(log_format)                         # set format to what we wrote out
    console_handler.setLevel(logging.WARNING)                        # set level to WARNING
    logger.handlers = [console_handler]                              # add to list of handlers

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)                 # also saves log to file 'log_file'
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def_logger = init_logger()