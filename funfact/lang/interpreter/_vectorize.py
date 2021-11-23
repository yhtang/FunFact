#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from funfact.util.iterable import as_tuple
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import TranscribeInterpreter


class Vectorizer(TranscribeInterpreter):

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('outidx')

    def __init__(self, replicas: int):
        self.replicas = replicas
        self.vec_index = P.index(AbstractIndex(), bound=False, kron=False)

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return []

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        abstract._shape = as_tuple([*abstract.shape, self.replicas])
        if abstract.initializer is not None:
            if not callable(abstract.initializer):
                abstract.initializer = abstract.initializer[..., None]
        return []

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return []

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return []

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: P.indices, live_indices,
        keep_indices, **kwargs
    ):
        indices.items.append(self.vec_index)
        return []

    @as_payload
    def call(self, f: str, x: Tensorial, live_indices,
             keep_indices, **kwargs):
        return []

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, live_indices,
            keep_indices, **kwargs):
        return None

    @as_payload
    def neg(self, x: Numeric, live_indices,
            keep_indices, **kwargs):
        return None

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            keep_indices, **kwargs):
        return P.indices([*[P.index(i, bound=False, kron=False) for i in
                         live_indices], self.vec_index])

    @as_payload
    def tran(self, src: Numeric, indices: P.indices, live_indices,
             keep_indices, **kwargs):
        return []
