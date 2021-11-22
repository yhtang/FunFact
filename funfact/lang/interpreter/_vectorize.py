#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import Optional
from funfact.util.iterable import as_tuple
from funfact.lang._ast import _AST, _ASNode, Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from ._base import _deep_apply, TranscribeInterpreter


class Vectorizer():

    Tensorial = TranscribeInterpreter.Tensorial
    Numeric = TranscribeInterpreter.Numeric

    def literal(self, value: LiteralValue, **kwargs):
        pass

    def tensor(self, abstract: AbstractTensor, **kwargs):
        abstract._shape = as_tuple([*abstract.shape, self.replicas])
        if abstract.initializer is not None:
            if not callable(abstract.initializer):
                abstract.initializer = abstract.initializer[..., None]

    def index(self, item: AbstractIndex, bound: bool, **kwargs):
        pass

    def indices(self, items: AbstractIndex, **kwargs):
        pass

    def index_notation(
        self, tensor: P.tensor, indices: P.indices, **kwargs
    ):
        indices.items = tuple([*indices.items, self.vec_index])

    def call(self, f: str, x: Tensorial, **kwargs):
        pass

    def pow(self, base: Numeric, exponent: Numeric, **kwargs):
        pass

    def neg(self, x: Numeric, replicas, **kwargs):
        pass

    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices],
            **kwargs):
        pass

    def tran(self, src: Numeric, indices: P.indices, **kwargs):
        pass

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        if parent is None:
            node = copy.deepcopy(node)
        rule = getattr(self, node.name)
        rule(**node.fields)
        for name, value in node.fields_fixed.items():
            setattr(node, name, _deep_apply(self, value, node))
        return node

    def __init__(self, replicas: int):
        self.replicas = replicas
        self.vec_index = P.index(AbstractIndex(), bound=True)

    def __ror__(self, tsrex: _AST):
        return type(tsrex)(self(tsrex.root))
