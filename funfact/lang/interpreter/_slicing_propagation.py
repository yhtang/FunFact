#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import PreOrderTranscriber


class SlicingPropagator(PreOrderTranscriber):
    '''The slicing propagator analyzes which of the slices of the leafs
    and intermediate nodes should be computed to get the desired
    output at the root.'''

    Tensorial = PreOrderTranscriber.Tensorial
    Numeric = PreOrderTranscriber.Numeric

    def __init__(self, slices):
        self.slices = slices

    def __call__(self, node, parent=None):
        if parent is None:
            node.slices = self.slices
        return super().__call__(node, parent)

    def literal(self, value: LiteralValue, **kwargs):
        pass

    def tensor(self, abstract: AbstractTensor, **kwargs):
        pass

    def index(self, item: AbstractIndex, bound: bool, kron: bool, **kwargs):
        pass

    def indices(self, items: AbstractIndex, **kwargs):
        pass

    def index_notation(
        self, indexless: Numeric, indices: P.indices, slices, **kwargs
    ):
        indexless.slices = slices

    def call(self, f: str, x: Tensorial, slices, **kwargs):
        x.slices = slices

    def pow(self, base: Numeric, exponent: Numeric, slices, **kwargs):
        base.slices = slices
        exponent.slices = slices

    def neg(self, x: Numeric, slices, **kwargs):
        x.slices = slices

    def binary(self, lhs: Numeric, rhs: Numeric, oper: str, slices, **kwargs):
        lhs.slices = slices
        rhs.slices = slices

    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], slices, live_indices,
            **kwargs):
        slice_dict = dict(zip(live_indices, slices))
        lhs_slices = []
        for i in lhs.live_indices:
            try:
                lhs_slices.append(slice_dict[i])
            except KeyError:
                lhs_slices.append(slice(None))
        rhs_slices = []
        for i in rhs.live_indices:
            try:
                rhs_slices.append(slice_dict[i])
            except KeyError:
                rhs_slices.append(slice(None))
        lhs.slices = lhs_slices
        rhs.slices = rhs_slices
        if outidx is not None:
            outidx.slices = None

    def tran(self, src: Numeric, indices: P.indices, slices, **kwargs):
        src.slices = [
            slices[src.live_indices.index(i)] for i in indices.live_indices
        ]
