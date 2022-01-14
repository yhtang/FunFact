#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from ._base import (
    _as_payload,
    TranscribeInterpreter
)
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue


class IndexnessAnalyzer(TranscribeInterpreter):
    '''Determines indexness inheritance.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    as_payload = _as_payload(
        'indexed'
    )

    @as_payload
    def abstract_index_notation(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        return True

    @as_payload
    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        if isinstance(lhs, P.literal) is not isinstance(rhs, P.literal):
            return lhs.indexed or rhs.indexed
        else:
            return lhs.indexed and rhs.indexed

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return False

    @as_payload
    def tensor(self, decl: AbstractTensor, **kwargs):
        return False

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, kron: bool, **kwargs):
        return False

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return False

    @as_payload
    def indexed_tensor(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        return True

    @as_payload
    def call(self, f: str, x: P.Tensorial, **kwargs):
        return x.indexed

    @as_payload
    def neg(self, x: P.Numeric, **kwargs):
        return x.indexed

    @as_payload
    def elem(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        return False

    @as_payload
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], **kwargs
    ):
        return True

    @as_payload
    def tran(self, src: P.Numeric, indices: P.indices, **kwargs):
        return indices is not None

    @as_payload
    def abstract_dest(self, src: P.Numeric, indices: P.indices, **kwargs):
        return True
