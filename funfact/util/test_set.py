#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from .set import (
    ordered_union,
    ordered_intersect,
    ordered_setminus,
    ordered_symmdiff
)


def test_ordered_union():
    assert ordered_union([], []) == []
    assert ordered_union([], [1, 2]) == [1, 2]
    assert ordered_union([1, 2], []) == [1, 2]
    assert ordered_union([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert ordered_union([2, 1], [4, 3]) == [2, 1, 4, 3]
    assert ordered_union([2, 1], [1, 3]) == [2, 1, 3]
    assert ordered_union([1, 2], [1, 2, 3]) == [1, 2, 3]
    assert ordered_union([1, 2], [0, 1, 2]) == [1, 2, 0]
    assert ordered_union([1, 2, 3], [3, 2, 1]) == [1, 2, 3]
    assert ordered_union([1, 2, 3], [4, 3, 2]) == [1, 2, 3, 4]


def test_ordered_intersect():
    assert ordered_intersect([], []) == []
    assert ordered_intersect([], [1, 2]) == []
    assert ordered_intersect([1, 2], []) == []
    assert ordered_intersect([1, 2], [3, 4]) == []
    assert ordered_intersect([2, 1], [4, 3]) == []
    assert ordered_intersect([2, 1], [1, 3]) == [1]
    assert ordered_intersect([1, 2], [1, 2, 3]) == [1, 2]
    assert ordered_intersect([1, 2], [0, 1, 2]) == [1, 2]
    assert ordered_intersect([1, 2, 3], [3, 2, 1]) == [1, 2, 3]
    assert ordered_intersect([1, 2, 3], [4, 3, 2]) == [2, 3]


def test_ordered_setminus():
    assert ordered_setminus([], []) == []
    assert ordered_setminus([], [1, 2]) == []
    assert ordered_setminus([1, 2], []) == [1, 2]
    assert ordered_setminus([1, 2], [3, 4]) == [1, 2]
    assert ordered_setminus([2, 1], [4, 3]) == [2, 1]
    assert ordered_setminus([2, 1], [1, 3]) == [2]
    assert ordered_setminus([1, 2], [1, 2, 3]) == []
    assert ordered_setminus([1, 2], [0, 1, 2]) == []
    assert ordered_setminus([1, 2, 3], [3, 2, 1]) == []
    assert ordered_setminus([1, 2, 3], [4, 3, 2]) == [1]
    assert ordered_setminus([2, 3, 4], [1, 2]) == [3, 4]


def test_ordered_symmdiff():
    assert ordered_symmdiff([], []) == []
    assert ordered_symmdiff([], [1, 2]) == [1, 2]
    assert ordered_symmdiff([1, 2], []) == [1, 2]
    assert ordered_symmdiff([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert ordered_symmdiff([2, 1], [4, 3]) == [2, 1, 4, 3]
    assert ordered_symmdiff([2, 1], [1, 3]) == [2, 3]
    assert ordered_symmdiff([1, 2], [1, 2, 3]) == [3]
    assert ordered_symmdiff([1, 2], [0, 1, 2]) == [0]
    assert ordered_symmdiff([1, 2, 3], [3, 2, 1]) == []
    assert ordered_symmdiff([1, 2, 3], [4, 3, 2]) == [1, 4]
    assert ordered_symmdiff([2, 3, 4], [1, 2]) == [3, 4, 1]
