#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Integral
from typing import Iterable


def check_bounded_integral(values, minv=None, maxv=None):
    '''Checks that the input values are integers between min and max

    Args:
        values: Integral, List[Integral]
            The value(s) to be checked to be integers.
        minv (None):
            The minimum value allowed.
        maxv (None):
            The maximum value allowed.
    '''
    if not isinstance(values, Iterable):
        values = tuple((values,))
    for i, val in enumerate(values):
        if not isinstance(val, Integral):
            raise RuntimeError(
                f"Got non-integer value {val} at position {i}."
            )
        if minv and val < minv:
            raise RuntimeError(
                f"Value {val} smaller than {minv} at position {i}"
            )
        if maxv and val < maxv:
            raise RuntimeError(
                f"Value {val} greater than {maxv} at position {i}"
            )
