#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from .iterable import (
    as_namedtuple,
    as_tuple,
    flatten,
    flatten_if,
    flatten_dict,
    map_or_call,
)


def test_flatten():
    assert flatten([]) == ()
    assert flatten(()) == ()
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


@pytest.mark.parametrize('case', [
    ({}, ()),
    (dict(), ()),
    (dict(a=1, b=2), (1, 2)),
    (dict(a=1, b=2, c=dict(x=1.5, y=False)), (1, 2, 1.5, False)),
    ({1: 2, 3: 4}, (2, 4)),
    ({'a': 'b', True: 4}, ('b', 4)),
    ({1: dict(a=0), 2: {'b': 0}}, (0, 0))
])
def test_flatten_dict(case):
    input, output = case
    assert flatten_dict(input) == output


def test_map_or_call():
    assert tuple(map_or_call(range(3), lambda x: x**2)) == (0, 1, 4)
    assert tuple(map_or_call(['a', 'b'], dict(a=1, b=2))) == (1, 2)


def test_as_namedtuple():
    t = as_namedtuple('NT', a=1, b=2)
    assert hasattr(t, 'a')
    assert hasattr(t, 'b')
    assert tuple(t) == (1, 2)


def test_as_tuple():
    assert as_tuple(1) == (1,)
    assert as_tuple(True) == (True,)
    assert as_tuple(1.5) == (1.5,)
    assert as_tuple([1, 2, 3]) == (1, 2, 3)
    assert as_tuple((1, 2, 3)) == (1, 2, 3)
