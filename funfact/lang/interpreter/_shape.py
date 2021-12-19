#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractTensor


class ShapeAnalyzer(TranscribeInterpreter):
    '''The shape analyzer checks the shapes of the nodes in the AST.'''
    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('shape')

    @as_payload
    def noop(self, **kwargs):
        return None

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return abstract.shape

    @as_payload
    def index_notation(self, tensor: P.tensor, indices: P.indices,  **kwargs):
        return tensor.shape

    @as_payload
    def call(self, f: str, x: Tensorial, **kwargs):
        return x.shape

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        if base.shape:
            return base.shape
        else:
            return exponent.shape

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return x.shape

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            kron_indices, **kwargs):
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
    def tran(self, src: Numeric, live_indices, **kwargs):
        return tuple(src.shape[src.live_indices.index(i)]
                     for i in live_indices)
