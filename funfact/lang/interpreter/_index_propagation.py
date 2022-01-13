#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from typing import Optional
import numpy as np
from ._base import (
    _as_payload,
    dfs_filter,
    RewritingTranscriber
)
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from funfact.util.set import ordered_intersect, ordered_union, ordered_setminus


class IndexAnalyzer(RewritingTranscriber):
    '''Analyzes which of the indices survive in a tensor operations and does
    AST rewrite to replace certain operations with specialized Einstein
    operations and index renaming operations.'''

    as_payload = _as_payload(
        'live_indices', 'keep_indices', 'kron_indices', 'shape'
    )

    def abstract_index_notation(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        if tensor.live_indices:
            '''triggers renaming of free indices:
                for new, old in zip(indices, live_indices):
                    dfs_replace(old, new)
            '''
            live_new = [i.item for i in indices.items]
            live_old = tensor.live_indices

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
            for n in dfs_filter(lambda n: n.name == 'index', tensor):
                i = n.item
                if i not in live_old and i in live_new:
                    index_map[i] = AbstractIndex()

            for n in dfs_filter(lambda n: n.name == 'index', tensor):
                n.item = index_map.get(n.item, n.item)

            return self(tensor)
        else:
            '''creates index notation'''
            return self(P.indexed_tensor(tensor, indices), depth=0)

    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        def _0(node):
            return self(node, depth=0)

        def _X(node):
            return self(node)

        '''Replace by einop if both operands are indexed.'''
        if operator == 'matmul':
            i, j, k = [
                P.index(AbstractIndex(), bound=False, kron=False)
                for _ in range(3)
            ]
            return P.ein(
                _0(P.abstract_index_notation(lhs, _X(P.indices([i, j])))),
                _0(P.abstract_index_notation(rhs, _X(P.indices([j, k])))),
                precedence=precedence,
                reduction='sum',
                pairwise='multiply',
                outidx=None
            )
        elif operator == 'kron':
            i, j = [
                P.index(AbstractIndex(), bound=False, kron=True)
                for _ in range(2)
            ]
            return P.ein(
                _0(P.abstract_index_notation(lhs, _X(P.indices([i, j])))),
                _0(P.abstract_index_notation(rhs, _X(P.indices([i, j])))),
                precedence=precedence,
                reduction='sum',
                pairwise='multiply',
                outidx=None
            )
        elif lhs.live_indices is not None and rhs.live_indices is not None:
            return P.ein(
                lhs, rhs,
                precedence=precedence,
                reduction='sum',
                pairwise=operator,
                outidx=None
            )
        else:
            return P.elem(lhs, rhs, precedence, operator)

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return None, None, None, ()

    @as_payload
    def tensor(self, decl: AbstractTensor, **kwargs):
        return None, None, None, decl.shape

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, kron: bool, **kwargs):
        return (
            [item],
            [item] if bound or kron else [],
            [item] if kron else [],
            None
        )

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return (
            list(it.chain.from_iterable([i.live_indices for i in items])),
            list(it.chain.from_iterable([i.keep_indices for i in items])),
            list(it.chain.from_iterable([i.kron_indices for i in items])),
            None
        )

    @as_payload
    def indexed_tensor(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        return (
            indices.live_indices,
            indices.keep_indices,
            indices.kron_indices,
            tensor.shape
        )

    @as_payload
    def call(self, f: str, x: P.Tensorial, **kwargs):
        return x.live_indices, x.keep_indices, x.kron_indices, x.shape

    @as_payload
    def neg(self, x: P.Numeric, **kwargs):
        return x.live_indices, x.keep_indices, x.kron_indices, x.shape

    @as_payload
    def elem(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **kwargs
    ):
        try:
            return None, None, None, np.broadcast_shapes(lhs.shape, rhs.shape)
        except RuntimeError:
            raise ValueError(
                f'Cannot perform elementwise operations on '
                f'tensors of incompatible shape {lhs.shape} and {rhs.shape}.'
            )

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
            live_indices, keep_indices, kron_indices = (
                implied_survival, lone_keep, kron
            )
        else:
            explicit_survival = outidx.live_indices
            explicit_kron = ordered_union(kron, outidx.kron_indices)
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
            live_indices, keep_indices, kron_indices = (
                explicit_survival, [], explicit_kron
            )

        dict_lhs = dict(zip(lhs.live_indices, lhs.shape))
        dict_rhs = dict(zip(rhs.live_indices, rhs.shape))

        print('lhs', lhs)
        print('dict_lhs', dict_lhs)

        print('rhs', rhs)
        print('dict_rhs', dict_rhs)

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

        return live_indices, keep_indices, kron_indices, tuple(shape)

    @as_payload
    def tran(self, src: P.Numeric, indices: P.indices, **kwargs):
        # if isinstance(node, P.tran) and isinstance(node.src, P.ein):
        #     '''override the `>>` behavior for einop nodes'''
        #     node.src.outidx = node.indices
        #     node = node.src
        return (
            indices.live_indices,
            indices.keep_indices,
            indices.kron_indices,
            tuple(src.shape[src.live_indices.index(i)]
                  for i in indices.live_indices)
        )
