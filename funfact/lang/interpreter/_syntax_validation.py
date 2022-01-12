#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from funfact.lang._ast import _ASNode, _AST, Primitives as P
from ._base import _deep_apply


class SyntaxValidator:

    def literal(self, node: _ASNode, parent: _ASNode):
        pass

    def tensor(self, node: _ASNode, parent: _ASNode):
        decl = node.decl
        ini = node.decl.initializer
        if ab.is_tensor(ini):
            if ini.shape != decl.shape:
                raise SyntaxError(
                    f'The shape {decl.shape} of tensor {decl} does not match '
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
        # if not isinstance(node.tensor, P.tensor):
        #     raise SyntaxError(
        #         f'Index notation only applies to a tensor object, '
        #         f'got {node.tensor} instead.'
        #     )
        if len(node.indices.items) != node.tensor.value.ndim:
            raise SyntaxError(
                f'Number of indices in {node.indices.items} does not match '
                f'the rank of tensor {node.tensor}.'
            )

    def call(self, node: _ASNode, parent: _ASNode):
        pass

    def neg(self, node: _ASNode, parent: _ASNode):
        pass

    def binary():
        # TODO
        pass

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        for name, value in node.fields_fixed.items():
            _deep_apply(self, value, node)
        return getattr(self, node.name)(node, parent)

    def __ror__(self, tsrex: _AST):
        self(tsrex.root)
