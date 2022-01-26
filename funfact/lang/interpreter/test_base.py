#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from .._ast import primitive, Primitives as P, _ASNode
from ._base import (
    _as_payload,
    _emplace,
    ROOFInterpreter,
    TranscribeInterpreter,
    RewritingTranscriber,
    PayloadMerger,
    dfs,
    dfs_filter
)


def test_as_payload():
    @_as_payload('x')
    def f():
        return 1

    assert f() == ('x', 1)

    @_as_payload('x', 'y')
    def f():
        return 1, 2

    assert f() == {'x': 1, 'y': 2}


@pytest.mark.parametrize('test_case', [
    ({'x': 1, 'y': 2}, [('x', 1), ('y', 2)]),
    ([('x', 1), ('y', 2)], [('x', 1), ('y', 2)]),
    (('c', 3), [('c', 3)]),
    (None, []),
])
def test_emplace(test_case):

    payload, result = test_case

    class Node:
        pass

    node = Node()
    _emplace(node, payload)
    for attr, value in result:
        assert hasattr(node, attr)
        assert getattr(node, attr) == value

    with pytest.raises(TypeError):
        _emplace(node, 'abc')
    with pytest.raises(TypeError):
        _emplace(node, 123)


def test_merger():

    @primitive()
    def binode(left, right):
        pass

    tree1 = binode(binode(None, None), None)
    tree2 = binode(binode(None, None), None)

    tree1.x = 1
    tree1.left.y = 2

    tree2.a = 'a'
    tree2.left.b = 'b'

    tree = PayloadMerger()(tree1, tree2)

    assert tree.x == 1
    assert tree.left.y == 2
    assert tree.a == 'a'
    assert tree.left.b == 'b'


@primitive()
def binode(left, right, value):
    pass


@pytest.mark.parametrize('test_case', [
    (binode(None, None, 0), [0]),
    (binode(binode(None, None, 0), None, 1), [0, 1]),
    (binode(None, binode(None, None, 0), 1), [0, 1]),
    (binode(binode(None, None, 0), binode(None, None, 1), 2), [0, 1, 2]),
])
def test_dfs(test_case):
    tree, flat = test_case
    assert list(map(lambda n: n.value, dfs(tree))) == flat


@pytest.mark.parametrize('test_case', [
    (binode(None, None, 0), [0]),
    (binode(binode(None, None, -1), None, 1), [1]),
    (binode(None, binode(None, None, 0), -1), [0]),
    (binode(binode(None, None, 0), binode(None, None, -1), 2), [0, 2]),
])
def test_dfs_filter(test_case):
    tree, flat = test_case
    assert list(
        map(lambda n: n.value, dfs_filter(lambda n: n.value >= 0, tree))
    ) == flat


@pytest.mark.parametrize('intr', [
    ROOFInterpreter,
    TranscribeInterpreter,
    RewritingTranscriber
])
def test_abstract_interpreter(intr):
    for prim in filter(
        lambda a: isinstance(a, type) and issubclass(a, _ASNode),
        P.__dict__.values()
    ):
        assert hasattr(intr, prim.__name__)
