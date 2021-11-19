#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue


class ShapeAnalyzer(TranscribeInterpreter):
    '''The shape analyzer checks the shapes of the nodes in the AST.'''
    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('shape')

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return None

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return abstract.shape

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
        return None

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return None

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            **kwargs):
        return None
