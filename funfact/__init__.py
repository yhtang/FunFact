#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fun-Fact"""
from .lang import index, indices, tensor, special
from .lang._math import *  # noqa: F401, F403


__all__ = ['index', 'indices', 'tensor', 'special']


__version__ = '0.6'
__author__ = '''Yu-Hang "Maxin" Tang, Daan Camps, Elizaveta Rebrova'''
__maintainer__ = 'Yu-Hang "Maxin" Tang'
__email__ = 'Tang.Maxin@gmail.com'
__license__ = 'see LICENSE file'
__copyright__ = '''see COPYRIGHT file'''
