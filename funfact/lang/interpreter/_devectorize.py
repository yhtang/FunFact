#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import jax.numpy as np
from typing import Optional
from funfact.util.iterable import as_tuple, flatten
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import TranscribeInterpreter


class Devectorizer(TranscribeInterpreter):

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('outidx')

    def __init__(self, slice: int):
        self.slice = slice

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        return []

    @as_payload
    def tensor(self, abstract: AbstractTensor, **kwargs):
        return []

    @as_payload
    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        return []

    @as_payload
    def indices(self, items: AbstractIndex, **kwargs):
        return []

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: P.indices, live_indices,
        keep_indices, **kwargs
    ):
        # update indices.items w/o altering original
        items = copy.copy(indices.items)
        items.pop()
        indices.items = items
        # update tensor.abstract w/o altering original
        abstract = copy.copy(tensor.abstract)
        data = copy.copy(tensor.data)
        shape = [abstract.shape]
        shape.pop()
        abstract._shape = as_tuple(flatten(shape))
        if abstract.initializer is not None and \
           not callable(abstract.initializer):
            initializer = copy.copy(abstract.initializer)
            abstract.initializer = np.squeeze(initializer)
            data = np.squeeze(data)
        else:
            data = copy.copy(abstract.data)
            data = np.squeeze(data[..., self.slice])
        tensor.abstract = abstract
        tensor.data = data
        return []

    @as_payload
    def call(self, f: str, x: Tensorial, live_indices,
             keep_indices, **kwargs):
        return []

    @as_payload
    def pow(self, base: Numeric, exponent: Numeric, live_indices,
            keep_indices, **kwargs):
        return []

    @as_payload
    def neg(self, x: Numeric, live_indices,
            keep_indices, **kwargs):
        return []

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            **kwargs):
        return P.indices([P.index(i, bound=False, kron=False) for i in
                         live_indices[:-1]])

    @as_payload
    def tran(self, src: Numeric, indices: P.indices, live_indices,
             keep_indices, **kwargs):
        return []
