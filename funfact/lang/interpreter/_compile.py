#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
from typing import Optional
import numpy as np
from ._base import (
    _as_payload,
    dfs_filter,
    RewritingTranscriber,
)
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from funfact.util.iterable import as_namedtuple
from funfact.util.set import ordered_intersect, ordered_union, ordered_setminus


def _add_attr(node, **kwargs):
    for key, val in kwargs.items():
        setattr(node, key, val)
    return node


class Compiler(RewritingTranscriber):
    '''Analyzes which of the indices survive in a tensor operations and does
    AST rewrite to replace certain operations with specialized Einstein
    operations and index renaming operations.'''

    as_payload = _as_payload(
        'live_indices', 'keep_indices', 'kron_indices', 'shape'
    )

    def abstract_index_notation(
        self, tensor: P.Numeric, indices: P.indices, **kwargs
    ):
        _0 = lambda node: self(node, depth=0)  # noqa: E731
        _X = lambda node: self(node)           # noqa: E731
        if tensor.indexed:
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

            return _X(tensor)  # recursively rebuild all live indices
        else:
            # only rebuild own indices
            return _add_attr(
                _0(P.indexed_tensor(tensor, indices)),
                indexed=True
            )

    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        indexed: bool, **kwargs
    ):
        '''Replace by einop if both operands are indexed.'''
        _0 = lambda node: self(node, depth=0)  # noqa: E731
        _X = lambda node: self(node)           # noqa: E731
        if operator == 'matmul':
            i, j, k = [
                P.index(AbstractIndex(), bound=False, kron=False)
                for _ in range(3)
            ]
            return _add_attr(
                P.ein(
                    _0(P.abstract_index_notation(lhs, _X(P.indices([i, j])))),
                    _0(P.abstract_index_notation(rhs, _X(P.indices([j, k])))),
                    precedence=precedence,
                    reduction='sum',
                    pairwise='multiply',
                    outidx=None
                ),
                indexed=indexed
            )
        elif operator == 'kron':
            i, j = [
                P.index(AbstractIndex(), bound=False, kron=True)
                for _ in range(2)
            ]
            return _add_attr(
                P.ein(
                    _0(P.abstract_index_notation(lhs, _X(P.indices([i, j])))),
                    _0(P.abstract_index_notation(rhs, _X(P.indices([i, j])))),
                    precedence=precedence,
                    reduction='sum',
                    pairwise='multiply',
                    outidx=None
                ),
                indexed=indexed
            )
        elif indexed:
            return _add_attr(
                P.ein(
                    lhs, rhs,
                    precedence=precedence,
                    reduction='sum',
                    pairwise=operator,
                    outidx=None
                ),
                indexed=indexed
            )
        else:
            return _add_attr(
                P.elem(lhs, rhs, precedence, operator), indexed=indexed
            )

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
        src_live = as_namedtuple(
            'src_live', lhs=lhs.live_indices or [], rhs=rhs.live_indices or []
        )
        src_keep = as_namedtuple(
            'src_keep', lhs=lhs.keep_indices or [], rhs=rhs.keep_indices or []
        )
        src_kron = as_namedtuple(
            'src_kron', lhs=lhs.kron_indices or [], rhs=rhs.kron_indices or []
        )

        live = ordered_union(src_live.lhs, src_live.rhs)
        keep = ordered_union(src_keep.lhs, src_keep.rhs)
        kron = ordered_union(src_kron.lhs, src_kron.rhs)
        if outidx is None:
            free_l = ordered_setminus(src_live.lhs, src_live.rhs)
            free_r = ordered_setminus(src_live.rhs, src_live.lhs)
            free = ordered_union(free_l, free_r)
            repeated = ordered_intersect(src_live.lhs, src_live.rhs)
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

        dict_lhs = dict(zip(src_live.lhs, lhs.shape))
        dict_rhs = dict(zip(src_live.rhs, rhs.shape))

        for i in src_live.lhs:
            if i in src_live.rhs and i not in kron_indices:
                if dict_lhs[i] != dict_rhs[i]:
                    raise SyntaxError(
                        f'Dimension of elementwise index {i} on left-hand side'
                        f' ({dict_lhs[i]}) does not match dimension of '
                        f'right-hand side ({dict_rhs[i]}).'
                    )

        shape = []
        for i in live_indices:
            if i in src_live.lhs and i in src_live.rhs:
                if i in kron_indices:
                    shape.append(dict_lhs[i]*dict_rhs[i])
                else:
                    shape.append(dict_lhs[i])
            elif i in src_live.lhs:
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

    def abstract_dest(self, src: P.Numeric, indices: P.indices, **kwargs):
        if isinstance(src, P.ein):
            '''override the `>>` behavior for einop nodes'''
            src.outidx = indices
            return src
        else:
            return P.tran(src, indices)
