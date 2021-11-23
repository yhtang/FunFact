#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .lang import index, indices, tensor, template, _0, _1, delta  # noqa: F401
from .lang._math import *  # noqa: F401, F403
from ._context import (  # noqa: F401
    default_mode,
    eager_mode,
    lazy_mode,
    set_eagerness,
    get_eagerness,
    push_eagerness,
    pop_eagerness
)


__version__ = '0.7a2'
__author__ = '''Yu-Hang "Maxin" Tang, Daan Camps, Elizaveta Rebrova'''
__maintainer__ = 'Yu-Hang "Maxin" Tang, Daan Camps'
__email__ = 'Tang.Maxin@gmail.com'
__license__ = 'see LICENSE file'
__copyright__ = '''see COPYRIGHT file'''
