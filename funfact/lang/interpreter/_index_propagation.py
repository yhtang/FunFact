#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from numbers import Real
from typing import Optional
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._tensor import AbstractIndex, AbstractTensor
from funfact.util.set import ordered_symmdiff


class IndexPropagator(TranscribeInterpreter):
    '''The index propagator analyzes which of the indices survive in a
    contraction of two tensors and passes them onto the parent node.'''

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('live_indices')

    @as_payload
    def scalar(self, value: Real, **kwargs):
        return []

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return []

    @as_payload
    def index(self, item: AbstractIndex, **kwargs):
        return [item.symbol]

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return list(it.chain.from_iterable([i.live_indices for i in items]))

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: P.indices, **kwargs
    ):
        return indices.live_indices

    @as_payload
    def call(self, f: str, x: Tensorial, **kwargs):
        return x.live_indices

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        return base.live_indices + exponent.live_indices

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return x.live_indices

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], **kwargs):
        if outidx is not None:
            return outidx.live_indices
        else:
            return ordered_symmdiff(lhs.live_indices, rhs.live_indices)
