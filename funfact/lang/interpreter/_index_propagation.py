#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from typing import Optional
from ._base import dfs_filter, TranscribeInterpreter
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from funfact.util.set import ordered_intersect, ordered_union, ordered_setminus


class IndexPropagator(TranscribeInterpreter):
    '''Analyzes which of the indices survive in a tensor operations and does
    AST rewrite to replace certain operations with specialized Einstein
    operations and index renaming operations.'''

    _traversal_order = TranscribeInterpreter.TraversalOrder.POST

    as_payload = TranscribeInterpreter.as_payload(
        'live_indices', 'keep_indices', 'kron_indices'
    )

    def __call__(self, node, parent=None):

        node = super().__call__(node, parent)

        if isinstance(node, P.binary):
            '''Replace by einop if both operands are indexed.'''
            if node.lhs.live_indices and node.rhs.live_indices:
                node = P.ein(
                    node.lhs, node.rhs, precedence=node.precedence,
                    reduction='sum', pairwise=node.oper, outidx=None
                )
                TranscribeInterpreter.emplace(
                    node,
                    getattr(self, node.name)(**node.fields)
                )

        if isinstance(node, P.index_notation) and node.indexless.live_indices:
            '''Triggers renaming of free indices:
            for new, old in zip(node.indices, node.live_indices):
                dfs_replace(old, new)
            '''
            live_new = [i.item for i in node.indices.items]
            live_old = node.indexless.live_indices

            if len(live_new) != len(live_old):
                raise SyntaxError(
                    f'Incorrect number of indices. '
                    f'Expects {len(live_old)}, '
                    f'got {len(live_new)}.'
                )

            index_map = dict(zip(live_old, live_new))
            # If a 'new' live index is already used as a dummy one,
            # replace the dummy usage with an anonymous index to avoid
            # conflict.
            node = node.indexless
            for n in dfs_filter(lambda n: n.name == 'index', node):
                i = n.item
                if i not in live_old and i in live_new:
                    index_map[i] = AbstractIndex()

            for n in dfs_filter(lambda n: n.name == 'index', node):
                n.item = index_map.get(n.item, n.item)

            node = super().__call__(node, parent)  # rebuild live indices

        if isinstance(node, P.tran) and isinstance(node.src, P.ein):
            '''override the `>>` behavior for einop nodes'''
            node.src.outidx = node.indices
            node = node.src

        return node

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return [], [], []

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return [], [], []

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, kron: bool, **kwargs):
        return [item], [item] if bound or kron else [], [item] if kron else []

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return (
            list(it.chain.from_iterable([i.live_indices for i in items])),
            list(it.chain.from_iterable([i.keep_indices for i in items])),
            list(it.chain.from_iterable([i.kron_indices for i in items]))
        )

    @as_payload
    def index_notation(
        self, indexless: P.Numeric, indices: P.indices, **kwargs
    ):
        return indices.live_indices, indices.keep_indices, indices.kron_indices

    @as_payload
    def call(self, f: str, x: P.Tensorial, **kwargs):
        return x.live_indices, x.keep_indices, x.kron_indices

    @as_payload
    def neg(self, x: P.Numeric, **kwargs):
        return x.live_indices, x.keep_indices, x.kron_indices

    @as_payload
    def matmul(self, lhs: P.Numeric, rhs: P.Numeric, **kwargs):
        return [], [], []

    @as_payload
    def binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, oper: str,
        **kwargs
    ):
        return [], [], []

    @as_payload
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], **kwargs
    ):
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
        kron = ordered_union(lhs.kron_indices, rhs.kron_indices)
        if outidx is None:
            free_l = ordered_setminus(lhs.live_indices, rhs.live_indices)
            free_r = ordered_setminus(rhs.live_indices, lhs.live_indices)
            free = ordered_union(free_l, free_r)
            repeated = ordered_intersect(lhs.live_indices, rhs.live_indices)
            bound = ordered_setminus(keep, free)
            lone_keep = ordered_setminus(keep, repeated)
            implied_survival = free_l + bound + free_r
            return implied_survival, lone_keep, kron
        else:
            explicit_survival = outidx.live_indices
            explicit_kron = outidx.kron_indices
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
            return explicit_survival, [], explicit_kron

    @as_payload
    def tran(self, src: P.Numeric, indices: P.indices, **kwargs):
        return indices.live_indices, indices.keep_indices, indices.kron_indices
