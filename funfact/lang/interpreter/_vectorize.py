#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import TranscribeInterpreter


class _VectorizerBase(TranscribeInterpreter):

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    def literal(self, value: LiteralValue, **kwargs):
        return []

    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return []

    def index_notation(
        self, indexless: P.Numeric, indices: P.indices, **kwargs
    ):
        return []

    def call(self, f: str, x: P.Tensorial, **kwargs):
        return []

    def neg(self, x: P.Numeric, **kwargs):
        return []

    def matmul(self, lhs: P.Numeric, rhs: P.Numeric, **kwargs):
        return []

    def kron(self, lhs: P.Numeric, rhs: P.Numeric, **kwargs):
        return []

    def binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, oper: str,
        **kwargs
    ):
        return []

    def tran(self, src: P.Numeric, indices: P.indices, **kwargs):
        return []


class LeafVectorizer(_VectorizerBase):

    as_payload = TranscribeInterpreter.as_payload

    def __init__(self, replicas: int, vec_index: P.index):
        self.replicas = replicas
        self.vec_index = vec_index

    @as_payload('abstract')
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return abstract.vectorize(self.replicas)

    @as_payload('items')
    def indices(self, items: AbstractIndex, **kwargs):
        return (*items, self.vec_index)

    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], **kwargs
    ):
        return []


class EinopVectorizer(_VectorizerBase):

    as_payload = TranscribeInterpreter.as_payload

    def __init__(self, vec_index: P.index):
        self.vec_index = vec_index

    def tensor(self, abstract: AbstractTensor, **kwargs):
        return []

    def indices(self, items: AbstractIndex, **kwargs):
        return []

    @as_payload('outidx')
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], live_indices, **kwargs
    ):
        return P.indices([
            *[P.index(i, bound=False, kron=False) for i in live_indices if i != self.vec_index.item],
            # *[P.index(i, bound=False, kron=False) for i in live_indices],
            self.vec_index
        ])
