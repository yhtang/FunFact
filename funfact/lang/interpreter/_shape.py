#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import _as_payload, TranscribeInterpreter


class ShapeAnalyzer(TranscribeInterpreter):
    '''The shape analyzer checks the shapes of the nodes in the AST.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    as_payload = _as_payload('shape')

    def abstract_index_notation(
        self, tensor: P.Numeric, indices: P.indices,  **kwargs
    ):
        raise NotImplementedError()

    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        raise NotImplementedError()

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return ()

    @as_payload
    def tensor(self, decl: AbstractTensor, **kwargs):
        return decl.shape

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return None

    @as_payload
    def indices(self, items: Tuple[P.index], **kwargs):
        return None

    @as_payload
    def indexed_tensor(
        self, tensor: P.Numeric, indices: P.indices,  **kwargs
    ):
        return tensor.shape

    @as_payload
    def call(self, f: str, x: P.Tensorial, **kwargs):
        return x.shape

    @as_payload
    def neg(self, x: P.Numeric, **kwargs):
        return x.shape

    @as_payload
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], live_indices, kron_indices,
        **kwargs
    ):
        dict_lhs = dict(zip(lhs.live_indices, lhs.shape))
        dict_rhs = dict(zip(rhs.live_indices, rhs.shape))

        for i in lhs.live_indices:
            if i in rhs.live_indices and i not in kron_indices:
                if dict_lhs[i] != dict_rhs[i]:
                    raise SyntaxError(
                        f'Dimension of elementwise index {i} on left-hand side'
                        f' ({dict_lhs[i]}) does not match dimension of '
                        f'right-hand side ({dict_rhs[i]}).'
                    )

        shape = []
        for i in live_indices:
            if i in lhs.live_indices and i in rhs.live_indices:
                if i in kron_indices:
                    shape.append(dict_lhs[i]*dict_rhs[i])
                else:
                    shape.append(dict_lhs[i])
            elif i in lhs.live_indices:
                shape.append(dict_lhs[i])
            else:
                shape.append(dict_rhs[i])
        return tuple(shape)

    @as_payload
    def tran(self, src: P.Numeric, live_indices, **kwargs):
        return tuple(src.shape[src.live_indices.index(i)]
                     for i in live_indices)
