#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
from ._base import ROOFInterpreter
from funfact.lang._ast import _ASNode, _AST
from funfact.lang.interpreter._base import _deep_apply
from ._einop import _einop


class ElementEvaluator(ROOFInterpreter):

    @staticmethod
    def _binary_operator(reduction, pairwise, lhs, rhs, spec):
        return _einop(spec, lhs, rhs, reduction, pairwise)

    def scalar(self, value, **kwargs):
        return value

    def tensor(self, abstract, data, **kwargs):
        return data

    def index(self, item, mustkeep, **kwargs):
        return None

    def indices(self, items, **kwargs):
        return None

    def index_notation(self, tensor, indices, **kwargs):
        return tensor

    def call(self, f, x, **kwargs):
        return getattr(np, f)(x)

    def pow(self, base, exponent, **kwargs):
        return np.power(base, exponent)

    def neg(self, x, **kwargs):
        return -x

    def ein(self, lhs, rhs, precendence, reduction, pairwise, outidx, einspec,
            **kwargs):
        return self._binary_operator(reduction, pairwise, lhs, rhs, einspec)

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        fields_fixed = {
            name: _deep_apply(self, value, node)
            for name, value in node.fields_fixed.items()
        }
        rule = getattr(self, node.name)
        return rule(**fields_fixed, **node.fields_payload)

    def __ror__(self, tsrex: _AST):
        return self(tsrex.root)
