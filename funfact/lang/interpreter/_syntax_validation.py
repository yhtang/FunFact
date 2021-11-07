#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang._ast import _ASNode, _AST, Primitives as P
from ._base import _deep_apply


class SyntaxValidator:
    '''A ROOF (Read-Only On-the-Fly) interpreter traverses an AST for one pass
    and produces the final outcome without altering the AST. Intermediates are
    passed as return values between the traversing levels. Its primitive rules
    may still accept a 'payload' argument, which could be potentially produced
    by another transcribe interpreter.'''

    def scalar(self, node: _ASNode, parent: _ASNode):
        pass

    def tensor(self, node: _ASNode, parent: _ASNode):
        pass

    def index(self, node: _ASNode, parent: _ASNode):
        pass

    def index_notation(self, node: _ASNode, parent: _ASNode):
        if not isinstance(node.tensor, P.tensor):
            raise SyntaxError(
                f'Index notation only applies to a tensor object, '
                f'got {node.tensor} instead.'
            )

    def call(self, node: _ASNode, parent: _ASNode):
        pass

    def pow(self, node: _ASNode, parent: _ASNode):
        pass

    def neg(self, node: _ASNode, parent: _ASNode):
        pass

    def mul(self, node: _ASNode, parent: _ASNode):
        pass

    def div(self, node: _ASNode, parent: _ASNode):
        pass

    def add(self, node: _ASNode, parent: _ASNode):
        pass

    def sub(self, node: _ASNode, parent: _ASNode):
        pass

    def let(self, node: _ASNode, parent: _ASNode):
        pass

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        for name, value in node.fields_fixed.items():
            _deep_apply(self, value, node)
        return getattr(self, node.name)(node, parent)

    def __ror__(self, tsrex: _AST):
        self(tsrex.root)
