#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import (
    AbstractIndex, AbstractTensor, LiteralValue, ParametrizedAbstractTensor
)
from funfact.util.iterable import as_namedtuple
from ._base import _as_payload, TranscribeInterpreter


class EinopCompiler(TranscribeInterpreter):
    '''The EinopCompiler generates the specifications to run the binary Einstein
    operations and single tensor transpositions in the Evaluator.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.DEPTH

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

    def parametrized_tensor(self, decl: ParametrizedAbstractTensor, **kwargs):
        return []

    def tensor(self, decl: AbstractTensor, **kwargs):
        return []

    def ellipsis(self, **kwargs):
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

    def elem(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        return []

    @_as_payload('einspec')
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], live_indices, kron_indices,
        **kwargs
    ):
        lhs_live = lhs.live_indices or []
        rhs_live = rhs.live_indices or []
        lhs_kron = lhs.kron_indices or []
        rhs_kron = rhs.kron_indices or []

        # move all surviving indices to the front and sort the rest
        all_indices = live_indices + sorted(
            (set(lhs_live) | set(rhs_live)) - set(live_indices or [])
        )
        kron_set = set(lhs_kron) | set(rhs_kron)

        # reorder lhs and rhs following the order in all indices
        tran_lhs = np.argsort([all_indices.index(i) for i in lhs_live])
        tran_rhs = np.argsort([all_indices.index(i) for i in rhs_live])
        if np.all(tran_lhs == np.arange(len(lhs_live))):
            tran_lhs = []
        if np.all(tran_rhs == np.arange(len(rhs_live))):
            tran_rhs = []

        # Determine expansion positions to align the contraction, Kronecker,
        # elementwise, and outer product indices
        p_out, p_lhs, p_rhs = 0, 0, 0
        index_lhs = []
        index_rhs = []
        ax_contraction = []
        newaxis, colon = None, slice(None)

        for i in all_indices:
            if i not in live_indices:  # contracting index
                ax_contraction.append(p_out)
                index_lhs.append(colon)
                index_rhs.append(colon)
                p_lhs += 1
                p_rhs += 1
                p_out += 1
            else:  # non-contracting index
                if i in kron_set:
                    index_lhs += [colon, newaxis]
                    index_rhs += [newaxis, colon]
                    p_lhs += 1
                    p_rhs += 1
                    p_out += 2
                else:
                    if i in lhs_live and i in rhs_live:
                        index_lhs.append(colon)
                        index_rhs.append(colon)
                        p_lhs += 1
                        p_rhs += 1
                        p_out += 1
                    elif i in lhs_live:
                        index_lhs.append(colon)
                        index_rhs.append(newaxis)
                        p_lhs += 1
                        p_out += 1
                    elif i in rhs_live:
                        index_lhs.append(newaxis)
                        index_rhs.append(colon)
                        p_rhs += 1
                        p_out += 1

        return as_namedtuple(
            'einspec',
            op_reduce=reduction,
            op_elementwise=pairwise,
            tran_lhs=tuple(tran_lhs),
            tran_rhs=tuple(tran_rhs),
            index_lhs=tuple(index_lhs),
            index_rhs=tuple(index_rhs),
            ax_contraction=tuple(ax_contraction)
        )

    @_as_payload('transpec')
    def tran(self, src: P.Numeric, indices: P.indices, live_indices, **kwargs):
        return as_namedtuple(
            'transpec',
            order=[src.live_indices.index(i) for i in indices.live_indices]
        )

    def abstract_dest(self, src: P.Numeric, indices: P.indices, **kwargs):
        raise NotImplementedError()
