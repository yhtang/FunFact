#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from typing import Optional
from ._base import TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from funfact.util.set import ordered_intersect, ordered_union, ordered_setminus


class IndexPropagator(TranscribeInterpreter):
    '''The index propagator analyzes which of the indices survive in a
    contraction of two tensors and passes them onto the parent node.'''

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload(
        'live_indices', 'keep_indices'
    )

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return [], []

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return [], []

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, kron: bool, **kwargs):
        return [item], [item] if bound else []

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
        live = ordered_union(lhs.live_indices, rhs.live_indices)
        keep = ordered_union(lhs.keep_indices, rhs.keep_indices)
        if outidx is None:
            free_l = ordered_setminus(lhs.live_indices, rhs.live_indices)
            free_r = ordered_setminus(rhs.live_indices, lhs.live_indices)
            free = ordered_union(free_l, free_r)
            repeated = ordered_intersect(lhs.live_indices, rhs.live_indices)
            bound = ordered_setminus(keep, free)
            lone_keep = ordered_setminus(keep, repeated)
            implied_survival = free_l + bound + free_r
            return implied_survival, lone_keep
        else:
            explicit_survival = outidx.live_indices
            for i in keep:
                if i not in explicit_survival:
                    raise SyntaxError(
                        f'Index {i} is marked as non-reducing, yet is missing '
                        f'from the explicit output list {explicit_survival}.'
                    )
            for i in explicit_survival:
                if i not in live:
                    raise SyntaxError(
                        f'Explicitly specified index {i} does not'
                        f'existing in the operand indices list {live}.'
                    )
            return explicit_survival, []

    @as_payload
    def tran(self, src: Numeric, indices: P.indices, **kwargs):
        return indices.live_indices, indices.keep_indices
