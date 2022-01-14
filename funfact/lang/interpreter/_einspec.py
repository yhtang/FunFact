#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import _as_payload, TranscribeInterpreter


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
            return ''.join([self._map(i) for i in ids or []])
        except TypeError:
            return self._map(ids)


class EinsteinSpecGenerator(TranscribeInterpreter):
    '''The Einstein summation specification generator creates NumPy-style spec
    strings for tensor contraction operations.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    as_payload = _as_payload('einspec')

    def abstract_index_notation(
        self, tensor: P.Tensorial, indices: P.indices,  **kwargs
    ):
        raise NotImplementedError()

    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        raise NotImplementedError()

    def literal(self, value: LiteralValue, **kwargs):
        return []

    def tensor(self, decl: AbstractTensor, **kwargs):
        return []

    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return []

    def indices(self, items: Tuple[P.index], **kwargs):
        return []

    def indexed_tensor(
        self, tensor: P.Tensorial, indices: P.indices,  **kwargs
    ):
        return []

    def call(self, f: str, x: P.Tensorial, **kwargs):
        return []

    def neg(self, x: P.Numeric, **kwargs):
        return []

    @as_payload
    def elem(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        return []

    @as_payload
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], live_indices, kron_indices,
        **kwargs
    ):
        map = IndexMap()
        return f'{map(lhs.live_indices)},{map(rhs.live_indices)}'\
               f'->{map(live_indices)}|{map(kron_indices)}'

    @as_payload
    def tran(self, src: P.Numeric, indices: P.indices, live_indices, **kwargs):
        map = IndexMap()
        return f'{map(src.live_indices)}->{map(live_indices)}'

    @as_payload
    def abstract_dest(self, src: P.Numeric, indices: P.indices, **kwargs):
        raise NotImplementedError()
