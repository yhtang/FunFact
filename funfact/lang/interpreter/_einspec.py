#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue


class IndexMap:
    def __init__(self):
        self._index_map = {}

    def _map(self, idx):
        try:
            return self._index_map[idx]
        except KeyError:
            self._index_map[idx] = chr(97 + len(self._index_map))
            return self._index_map[idx]

    def __call__(self, ids):
        try:
            return ''.join([self._map(i) for i in ids])
        except TypeError:
            return self._map(ids)


class EinsteinSpecGenerator(TranscribeInterpreter):
    '''The Einstein summation specification generator creates NumPy-style spec
    strings for tensor contraction operations.'''

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('einspec')

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return None

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return None

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return None

    @as_payload
    def indices(self, items: Tuple[P.index], **kwargs):
        return None

    @as_payload
    def index_notation(self, tensor: P.tensor, indices: P.indices,  **kwargs):
        return None

    @as_payload
    def call(self, f: str, x: Tensorial, **kwargs):
        return None

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        map = IndexMap()
        return f'{map(base.live_indices)},{map(exponent.live_indices)}'

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return None

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            **kwargs):
        map = IndexMap()
        return f'{map(lhs.live_indices)},{map(rhs.live_indices)}'\
               f'->{map(live_indices)}'

    @as_payload
    def tran(self, src: Numeric, indices: P.indices, live_indices, **kwargs):
        map = IndexMap()
        return f'{map(src.live_indices)}->{map(live_indices)}'
