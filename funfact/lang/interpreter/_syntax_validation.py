#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang._ast import _ASNode, _AST, Primitives as P
from ._base import _deep_apply
from funfact.util.typing import _is_tensor


class SyntaxValidator:
    '''A ROOF (Read-Only On-the-Fly) interpreter traverses an AST for one pass
    and produces the final outcome without altering the AST. Intermediates are
    passed as return values between the traversing levels. Its primitive rules
    may still accept a 'payload' argument, which could be potentially produced
    by another transcribe interpreter.'''

    def literal(self, node: _ASNode, parent: _ASNode):
        pass

    def tensor(self, node: _ASNode, parent: _ASNode):
        abst = node.abstract
        ini = node.abstract.initializer
        if _is_tensor(ini):
            if ini.shape != abst.shape:
                raise SyntaxError(
                    f'The shape {abst.shape} of tensor {abst} does not match '
                    f'its concrete-tensor initializer of {ini.shape}.'
                )

    def index(self, node: _ASNode, parent: _ASNode):
        i = node.item
        if i.bound and i.kron:
            raise SyntaxError(
                f'Index {i} should not simultaneous imply elementwise and '
                f'Kronecker product operations.'
            )

    def indices(self, node: _ASNode, parent: _ASNode):
        for i in node.items:
            if not isinstance(i, P.index):
                raise SyntaxError('Non-index item {i} found in indices.')

    def index_notation(self, node: _ASNode, parent: _ASNode):
        if not isinstance(node.tensor, P.tensor):
            raise SyntaxError(
                f'Index notation only applies to a tensor object, '
                f'got {node.tensor} instead.'
            )
        if len(node.indices.items) != node.tensor.value.ndim:
            raise SyntaxError(
                f'Number of indices in {node.indices.items} does not match '
                f'the rank of tensor {node.tensor}.'
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
