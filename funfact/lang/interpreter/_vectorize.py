#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import PostOrderTranscriber


class Vectorizer(PostOrderTranscriber):

    Tensorial = PostOrderTranscriber.Tensorial
    Numeric = PostOrderTranscriber.Numeric

    as_payload = PostOrderTranscriber.as_payload

    def __init__(self, replicas: int):
        self.replicas = replicas
        self.vec_index = P.index(AbstractIndex(), bound=False, kron=False)

    def literal(self, value: LiteralValue, **kwargs):
        return []

    @as_payload('abstract')
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return abstract.vectorize(self.replicas)

    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return []

    @as_payload('items')
    def indices(self, items: AbstractIndex, **kwargs):
        return (*items, self.vec_index)

    def index_notation(
        self, indexless: Numeric, indices: P.indices, live_indices,
        keep_indices, **kwargs
    ):
        return []

    def call(self, f: str, x: Tensorial, live_indices,
             keep_indices, **kwargs):
        return []

    def pow(self, base: Numeric, exponent: Numeric, live_indices,
            keep_indices, **kwargs):
        return []

    def neg(self, x: Numeric, live_indices,
            keep_indices, **kwargs):
        return []

    def binary(self, lhs: Numeric, rhs: Numeric, oper: str, **kwargs):
        return []

    @as_payload('outidx')
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            **kwargs):
        return P.indices([
            *[P.index(i, bound=False, kron=False) for i in live_indices],
            self.vec_index
        ])

    def tran(self, src: Numeric, indices: P.indices, live_indices,
             keep_indices, **kwargs):
        return []
