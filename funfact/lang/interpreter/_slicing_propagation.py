#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import Optional
from funfact.lang._ast import _AST, _ASNode, Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import _deep_apply, TranscribeInterpreter
from funfact.util.set import ordered_intersect, ordered_setminus, ordered_union


class SlicingPropagator():
    '''The slicing propagator analyzes which of the slices of the leafs
    and intermediate nodes should be computed to get the desired
    output at the root.'''

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    def literal(self, value: LiteralValue, slices, **kwargs):
        value.slices = None

    def tensor(self, abstract: AbstractTensor, slices, **kwargs):
        abstract.slices = None

    def index(self, item: AbstractIndex, bound: bool, slices, **kwargs):
        item.slices = None

    def indices(self, items: AbstractIndex, slices, **kwargs):
        for i in items:
            i.slices = None

    def index_notation(
        self, tensor: P.tensor, indices: P.indices, slices, **kwargs
    ):
        tensor.slices = slices
        indices.slices = None

    def call(self, f: str, x: Tensorial, slices, **kwargs):
        x.slices = slices

    def pow(self, base: Numeric, exponent: Numeric, slices, **kwargs):
        base.slices = slices
        exponent.slices = slices

    def neg(self, x: Numeric, slices, **kwargs):
        x.slices = slices

    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], slices, live_indices,
            **kwargs):
        indices_pl = ordered_intersect(live_indices, lhs.live_indices)
        indices_pr = ordered_intersect(live_indices, rhs.live_indices)
        indices_none = ordered_setminus(ordered_union(lhs.live_indices,
                                        rhs.live_indices), live_indices)
        lhs.slices = (*slices[:len(indices_pl)],
                      *((slice(None),) * len(indices_none)))
        rhs.slices = (*((slice(None),) * len(indices_none)),
                      *slices[-len(indices_pr):])

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        node = copy.copy(node)
        rule = getattr(self, node.name)
        if parent is None:
            node.slices = self.slices
        rule(**node.fields)
        for name, value in node.fields_fixed.items():
            setattr(node, name, _deep_apply(self, value, node))
        return node

    def __init__(self, slices):
        self.slices = slices

    def __ror__(self, tsrex: _AST):
        return type(tsrex)(self(tsrex.root))
