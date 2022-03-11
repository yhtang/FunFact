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
from funfact.lang._terminal import (
    AbstractIndex, AbstractTensor, LiteralValue, ParametrizedAbstractTensor
)
from funfact.util.iterable import as_namedtuple
from funfact.util.set import ordered_intersect, ordered_union, ordered_setminus


def _add_attr(node, **kwargs):
    for key, val in kwargs.items():
        setattr(node, key, val)
    return node


class TypeDeducer(RewritingTranscriber):
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
            # # TODO: check if tensors are 2D
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
            # # TODO: check if dimensionality of two tensors match
            indices = [
                P.index(AbstractIndex(), bound=False, kron=True)
                for _ in range(len(lhs.shape))
            ]
            return _add_attr(
                P.ein(
                    _0(P.abstract_index_notation(lhs, _X(P.indices(indices)))),
                    _0(P.abstract_index_notation(rhs, _X(P.indices(indices)))),
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
    def parametrized_tensor(self, decl: ParametrizedAbstractTensor, **kwargs):
        return None, None, None, decl.shape

    @as_payload
    def tensor(self, decl: AbstractTensor, **kwargs):
        return None, None, None, decl.shape

    @as_payload
    def ellipsis(self, **payload):
        return [P.ellipsis()], [], [], None

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
        # # TODO: check dimensionality match during multi dispatch of
        # # abstract_index_notation
        # if len(indices.live_indices) != len(tensor.shape):
        #     raise SyntaxError(
        #         f'Expects {len(tensor.shape)} indices, '
        #         f'got {len(indices.live_indices)}'
        #     )
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
        except ValueError:
            raise SyntaxError(
                f'Cannot perform elementwise operations on '
                f'tensors of incompatible shape {lhs.shape} and {rhs.shape}.'
            ) from None

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

        '''deduce survived indices'''
        live = ordered_union(src_live.lhs, src_live.rhs)
        keep = ordered_union(src_keep.lhs, src_keep.rhs)
        kron = ordered_union(src_kron.lhs, src_kron.rhs)
        free_l = ordered_setminus(src_live.lhs, src_live.rhs)
        free_r = ordered_setminus(src_live.rhs, src_live.lhs)
        free = ordered_union(free_l, free_r)
        bound = ordered_setminus(keep, free)
        implied_survival = free_l + bound + free_r
        if outidx is None:
            repeated = ordered_intersect(src_live.lhs, src_live.rhs)
            lone_keep = ordered_setminus(keep, repeated)
            lone_kron = ordered_setminus(kron, repeated)
            live_indices, keep_indices, kron_indices = (
                implied_survival, lone_keep, lone_kron
            )
        else:
            explicit_survival = outidx.live_indices
            if len(explicit_survival) < len(implied_survival):
                raise SyntaxError(
                    f'Expects at least {len(implied_survival)} explicit '
                    f'indices, got only {len(explicit_survival)}.'
                )
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
            lone_keep = ordered_setminus(keep, explicit_survival)
            lone_kron = ordered_setminus(kron, explicit_survival)
            live_indices, keep_indices, kron_indices = (
                explicit_survival, lone_keep, lone_kron
            )

        '''deduce shape of result tensor'''
        lhs_shape = dict(zip(src_live.lhs, lhs.shape))
        rhs_shape = dict(zip(src_live.rhs, rhs.shape))

        for i in src_live.lhs:
            if i in src_live.rhs and i not in kron:
                try:
                    np.broadcast_shapes(lhs_shape[i], rhs_shape[i])
                except ValueError:
                    raise SyntaxError(
                        f'Dimension of elementwise index {i} on left-hand side'
                        f' ({lhs_shape[i]}) cannot broadcast to that of '
                        f'the right-hand side ({rhs_shape[i]}).'
                    ) from None

        shape = []
        for i in live_indices:
            if i in src_live.lhs and i in src_live.rhs:
                if i in kron:
                    shape.append(lhs_shape[i] * rhs_shape[i])
                else:
                    shape.append(
                        *np.broadcast_shapes(lhs_shape[i], rhs_shape[i])
                    )
            elif i in src_live.lhs:
                shape.append(lhs_shape[i])
            else:
                shape.append(rhs_shape[i])
        return live_indices, keep_indices, kron_indices, tuple(shape)

    @as_payload
    def tran(self, src: P.Numeric, indices: P.indices, **kwargs):
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
