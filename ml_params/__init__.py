#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Root __init__
"""

import logging
from logging.config import dictConfig as _dictConfig
from os import path

import yaml

__author__ = "Samuel Marks"
__version__ = "0.0.17"


def get_logger(name=None):
    """
    Create a logger instance with the provided name, and default YAML config from this package

    :param name: Name of logger instance. Usually the module name with filename dot-appended. None gives root logger.
    :type name: Optional[str]

    :return: logger instance
    :rtype: ```logging.Logger```
    """
    with open(path.join(path.dirname(__file__), "_data", "logging.yml"), "rt") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    _dictConfig(data)
    return logging.getLogger(name=name)


root_logger = get_logger()
