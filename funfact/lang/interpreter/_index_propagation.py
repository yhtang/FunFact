#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from numbers import Real
from typing import Optional
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._tensor import AbstractIndex, AbstractTensor
from funfact.util.set import ordered_union, ordered_setminus


class IndexPropagator(TranscribeInterpreter):
    '''The index propagator analyzes which of the indices survive in a
    contraction of two tensors and passes them onto the parent node.'''

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload(
        'live_indices', 'keep_indices'
    )

    @as_payload
    def scalar(self, value: Real, **kwargs):
        return [], []

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return [], []

    @as_payload
    def index(self, item: AbstractIndex, mustkeep: bool, **kwargs):
        return [item.symbol], [item.symbol] if mustkeep else []

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return (
            list(it.chain.from_iterable([i.live_indices for i in items])),
            list(it.chain.from_iterable([i.keep_indices for i in items]))
        )

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: P.indices, **kwargs
    ):
        return indices.live_indices, indices.keep_indices

    @as_payload
    def call(self, f: str, x: Tensorial, **kwargs):
        return x.live_indices, x.keep_indices

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        return (
            base.live_indices + exponent.live_indices,
            base.keep_indices + exponent.keep_indices
        )

    @as_payload
    def neg(self, x: Numeric, **kwargs):
        return x.live_indices, x.keep_indices

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], **kwargs):
        '''
        ╔════════════╗
        ║     ╔══════╬═════╗
        ║   ╭─╫────╮ ║  L2 ║
        ║   │ ║K1  │ ║     ║
        ║   │ ║ ╭──┼─╫─╮   ║
        ║   ╰─╫─┼──╯ ║ │   ║
        ║     ║ │  K2║ │   ║
        ║ L1  ║ ╰────╫─╯   ║
        ╚═════╬══════╝     ║
              ╚════════════╝
        '''
        # indices marked as keep on either side should stay
        live_indices = ordered_union(lhs.live_indices, rhs.live_indices)
        keep_indices = ordered_union(lhs.keep_indices, rhs.keep_indices)
        if outidx is None:
            l_outer = ordered_setminus(lhs.live_indices, rhs.live_indices)
            r_outer = ordered_setminus(rhs.live_indices, lhs.live_indices)
            elementwise = ordered_setminus(
                ordered_union(lhs.keep_indices, rhs.keep_indices),
                ordered_union(l_outer, r_outer)
            )
            implied_out = l_outer + elementwise + r_outer
            return implied_out, []
        else:
            explicit_out = outidx.live_indices
            for i in keep_indices:
                if i not in explicit_out:
                    raise SyntaxError(
                        f'Index {i} is marked as non-reducing, yet is missing '
                        f'from the explicit output list {explicit_out}.'
                    )
            for i in explicit_out:
                if i not in live_indices:
                    raise SyntaxError(
                        f'Explicitly specified index {i} does not'
                        f'existing in the operand indices list {live_indices}.'
                    )
            return explicit_out, []
