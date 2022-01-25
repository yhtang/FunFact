#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dataclasses
from typing import Optional
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import _as_payload, TranscribeInterpreter


class Vectorizer(TranscribeInterpreter):

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    def __init__(self, replicas: int, vec_index: P.index,
                 append: bool = False):
        self.replicas = replicas
        self.vec_index = dataclasses.replace(vec_index, bound=True)
        self.append = append

    def _vectorize(self, *indices):
        if self.append:
            return (*indices, self.vec_index)
        else:
            return (self.vec_index, *indices)

    def abstract_index_notation(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        raise NotImplementedError()

    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        raise NotImplementedError()

    def literal(self, value: LiteralValue, **kwargs):
        return []

    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return []

    @_as_payload('decl')
    def tensor(self, decl: AbstractTensor, **kwargs):
        return decl.vectorize(self.replicas, self.append)

    @_as_payload('items')
    def indices(self, items: AbstractIndex, **kwargs):
        if self.append:
            return (*items, self.vec_index)
        return (self.vec_index, *items)

    def indexed_tensor(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        return []

    def call(self, f: str, x: P.Tensorial, **kwargs):
        return []

    def neg(self, x: P.Numeric, **kwargs):
        return []

    def elem(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        return []

    @_as_payload('outidx')
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], live_indices, **kwargs
    ):
        if outidx is not None:
            live_indices = [i.item for i in outidx.items]
        return P.indices(self._vectorize(*[
            P.index(i, bound=False, kron=False) for i in live_indices
            if i != self.vec_index.item
        ]))

    def tran(self, src: P.Numeric, indices: P.indices, **kwargs):
        return []

    def abstract_dest(self, src: P.Numeric, indices: P.indices, **kwargs):
        raise NotImplementedError()
