#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fun-Fact"""
from .backend import use, active_backend, available_backends
from .lang import index, indices, tensor, template, _0, _1, delta
from .lang._math import *  # noqa: F401, F403
from .model import Factorization
from .algorithm import factorize
from.optimizable import set_optimizable


__all__ = [
    'use',
    'active_backend',
    'available_backends',
    'index',
    'indices',
    'tensor',
    'template',
    '_0',
    '_1',
    'delta',
    'Factorization',
    'factorize',
    'set_optimizable',
]


__version__ = '0.7.0'
__author__ = '''Yu-Hang "Maxin" Tang, Daan Camps, Elizaveta Rebrova'''
__maintainer__ = 'Yu-Hang "Maxin" Tang'
__email__ = 'Tang.Maxin@gmail.com'
__license__ = 'see LICENSE file'
__copyright__ = '''see COPYRIGHT file'''
