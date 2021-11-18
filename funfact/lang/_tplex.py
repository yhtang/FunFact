#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from ._ast import Primitives as P
from .interpreter._base import _deep_apply


def _graft(node, select, repl):
    if select(node):
        return repl
    else:
        node = copy.copy(node)
        for name, value in node.fields_fixed.items():
            setattr(node, name, _deep_apply(
                _graft, value, select, repl
            ))
        return node


class Template:
    '''A template expression represents a reusable pattern. It must contain one
    special operand --- the tensor of all zeros, which serves as a
    placeholder for the actual tensor when the template is being
    instantiated.'''

    @staticmethod
    def _is_placeholder(n):
        return (
            isinstance(n, P.literal) and
            n.value.raw == 0
        )

    def __init__(self, tplex):
        self.tplex = tplex

    def __getitem__(self, indices):
        return type(self)(self.tplex[indices])

    def __matmul__(self, actual):
        return type(self.tplex)(
            _graft(
                self.tplex.root,
                self._is_placeholder,
                actual.root
            )
        )

    def __rmatmul__(self, actual):
        return self.__matmul__(actual)


def template(tplex):
    return Template(tplex)
