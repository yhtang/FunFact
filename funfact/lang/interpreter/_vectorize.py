#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import Optional
from funfact.util.iterable import as_tuple, flatten
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import AbstractIndex, LiteralValue
from ._base import TranscribeInterpreter


class Vectorizer(TranscribeInterpreter):

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    as_payload = TranscribeInterpreter.as_payload('outidx')

    def __init__(self, replicas: int):
        self.replicas = replicas
        self.vec_index = P.index(AbstractIndex(), bound=False, kron=False)

    @as_payload
    def noop(self):
        return []

    @as_payload
    def literal(self, value: LiteralValue, **kwargs):
        # TODO: vectorize indexed literals
        return []

    @as_payload
    def index_notation(
        self, tensor: P.tensor, indices: P.indices, live_indices,
        keep_indices, **kwargs
    ):
        # update indices.items w/o altering original
        items = copy.copy(indices.items)
        items.append(self.vec_index)
        indices.items = items
        # update tensor.abstract w/o altering original
        abstract = copy.copy(tensor.abstract)
        shape = [abstract.shape]
        shape.append(self.replicas)
        abstract._shape = as_tuple(flatten(shape))
        if abstract.initializer is not None:
            if not callable(abstract.initializer):
                initializer = copy.copy(abstract.initializer)
                abstract.initializer = initializer[..., None]
        tensor.abstract = abstract
        return []

    @as_payload
    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], live_indices,
            **kwargs):
        return P.indices([*[P.index(i, bound=False, kron=False) for i in
                         live_indices], self.vec_index])

    @as_payload
    def tran(self, src: Numeric, indices: P.indices, live_indices,
             keep_indices, **kwargs):
        # TODO: this looks suspicious
        return []
