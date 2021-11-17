#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from ._ast import _ASNode, _AST, Primitives


def test_primitive_decator():

    # create new primitive
    @Primitives.primitive(precedence=1)
    def prim(arg1, arg2):
        pass

    # check for basic properties
    assert isinstance(prim, type)
    assert issubclass(prim, _ASNode)
    assert hasattr(prim, 'name')
    assert hasattr(prim, 'precedence')
    assert hasattr(prim, 'fields')
    assert hasattr(prim, 'fields_fixed')
    assert hasattr(prim, 'fields_payload')

    # check for node instance properties
    node = prim('a', False)
    assert node.name == 'prim'
    assert node.precedence == 1
    assert 'arg1' in node.fields_fixed
    assert 'arg2' in node.fields_fixed
    assert node.arg1 == 'a'
    assert node.arg2 is False
    assert len(node.fields_payload) == 0
    # payload assignment
    node.payload1 = 1
    node.payload2 = 1.5
    assert len(node.fields_payload) == 2
    assert node.payload1 == 1
    assert node.payload2 == 1.5

    # per-node precedence
    @Primitives.primitive()
    def prim_nopred(precedence, arg1, arg2):
        pass

    assert not hasattr(prim_nopred, 'precedence')

    node = prim_nopred(2, 'a', True)
    assert node.name == 'prim_nopred'
    assert node.precedence == 2


def test_ast():

    @Primitives.primitive()
    def prim():
        pass

    # constract from scalar literal
    ast = _AST(1)
    assert hasattr(ast, 'root')
    assert ast.root.name == 'literal'
    # re-assign root
    ast.root = prim()
    assert ast.root.name == 'prim'

    # construct from ASNode
    assert _AST(prim()).root.name == 'prim'
