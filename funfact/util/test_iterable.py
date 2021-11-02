#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from .iterable import (
    as_namedtuple,
    flatten,
    flatten_if,
    flatten_dict,
    map_or_call,
)


def test_flatten():
    assert flatten([]) == ()
    assert flatten(tuple()) == ()
    assert flatten([1]) == (1,)
    assert flatten((1,)) == (1,)
    assert flatten((1, 2, 3)) == (1, 2, 3)
    assert flatten(range(5)) == (0, 1, 2, 3, 4)
    assert flatten([1, [2, 3]]) == (1, 2, 3)
    assert flatten([[1, 2], 3]) == (1, 2, 3)
    assert flatten([[1], [2], [3]]) == (1, 2, 3)
    assert flatten([range(2), range(2)]) == (0, 1, 0, 1)
    assert flatten([[[0]], []]) == (0,)
    assert flatten([[[0]], [range(1, 5)]]) == (0, 1, 2, 3, 4)


def test_flatten_if():
    assert flatten_if((1, (2,)), lambda i: isinstance(i, tuple)) == (1, 2)
    assert flatten_if((1, (2,)), lambda i: isinstance(i, list)) == (1, (2,))
