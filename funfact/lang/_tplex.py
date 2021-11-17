#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from ._ast import Primitives as P
from .interpreter._base import _deep_apply


def subtree_replace(node, pattern, repl):
    if pattern(node):
        return repl
    else:
        node = copy.copy(node)
        for name, value in node.fields_fixed.items():
            setattr(node, name, _deep_apply(
                subtree_replace, value, pattern, repl
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
            subtree_replace(
                self.tplex.root,
                self._is_placeholder,
                actual.root
            )
        )

    def __rmatmul__(self, actual):
        return self.__matmul__(actual)


def template(tplex):
    return Template(tplex)
