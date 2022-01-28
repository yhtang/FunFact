#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from funfact.util.iterable import as_namedtuple
from ._base import _as_payload, TranscribeInterpreter


class EinopCompiler(TranscribeInterpreter):
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
        # move all surviving indices to the front and sort the rest
        all_indices = live_indices + sorted(
            (set(lhs.live_indices) | set(rhs.live_indices)) - set(live_indices)
        )
        kron_indices = set(lhs.kron_indices) | set(rhs.kron_indices)

        # reorder lhs and rhs following the order in all indices
        tran_lhs = np.argsort([all_indices.index(i) for i in lhs.live_indices])
        tran_rhs = np.argsort([all_indices.index(i) for i in rhs.live_indices])

        # Determine expansion positions to align the contraction, Kronecker,
        # elementwise, and outer product indices
        p_out, p_lhs, p_rhs = 0, 0, 0
        index_lhs = []
        index_rhs = []
        ax_contraction = []
        # target_shape = []
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
                if i in kron_indices:
                    index_lhs += [colon, newaxis]
                    index_rhs += [newaxis, colon]
                    # target_shape.append(lhs.shape[p_lhs] * rhs.shape[p_rhs])
                    p_lhs += 1
                    p_rhs += 1
                    p_out += 2
                else:
                    if i in lhs.live_indices and i in rhs.live_indices:
                        # target_shape.append(
                        #     *ab.broadcast_shapes(
                        #         lhs.shape[p_lhs], rhs.shape[p_rhs]
                        #     )
                        # )
                        index_lhs.append(colon)
                        index_rhs.append(colon)
                        p_lhs += 1
                        p_rhs += 1
                        p_out += 1
                    elif i in lhs.live_indices:
                        index_lhs.append(colon)
                        index_rhs.append(newaxis)
                        # target_shape.append(lhs.shape[p_lhs])
                        p_lhs += 1
                        p_out += 1
                    elif i in rhs.live_indices:
                        index_lhs.append(newaxis)
                        index_rhs.append(colon)
                        # target_shape.append(rhs.shape[p_rhs])
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

    def tran(self, src: P.Numeric, indices: P.indices, live_indices, **kwargs):
        return []

    @as_payload
    def abstract_dest(self, src: P.Numeric, indices: P.indices, **kwargs):
        raise NotImplementedError()
