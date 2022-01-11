#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from ._ast import _ASNode, primitive, _AST


def test_make_primitive():

    # create new primitive
    @primitive(precedence=1)
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
    assert isinstance(node, prim)
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


def test_make_primitive_dynamic_precedence():
    # per-node precedence
    @primitive()
    def prim_dynamic_pred(precedence, arg1, arg2):
        pass

    assert not hasattr(prim_dynamic_pred, 'precedence')

    node = prim_dynamic_pred(2, 'a', True)
    assert node.name == 'prim_dynamic_pred'
    assert node.precedence == 2


def test_ast():
    ast = _AST(None)
    assert hasattr(ast, 'root')
    assert ast.root is None
    ast.root = None  # check if root is writable
