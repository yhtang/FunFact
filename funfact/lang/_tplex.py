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


class TemplateEx:
    '''A template expression represents a reusable pattern. It must contain one
    special operand --- a constant tensor of all 1s, which serves as a
    placeholder for the actual tensor when the template is being
    instantiated.'''

    @staticmethod
    def _is_placeholder(n):
        return (
            isinstance(n, P.index_notation) and
            n.tensor.abstract.symbol == '0'
        )

    def __init__(self, template):
        self.template = template

    def __getitem__(self, indices):
        return type(self)(self.template[indices])

    def __matmul__(self, insert):
        return type(self.template)(
            subtree_replace(
                self.template.root,
                self._is_placeholder,
                insert.root
            )
        )

    def __rmatmul__(self, insert):
        return self.__matmul__(insert)
