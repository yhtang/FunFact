#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
from abc import ABC, abstractmethod
import torch
from deap import gp


class FactorizationPrimitiveSet:

    class PrimitiveABC:

        @property
        @abstractmethod
        def action(self):
            pass

        def __init__(self, shape, *operands):
            self.operands = tuple(operands)

        def forward(self, grad=False):
            return self.action(*[o.forward(grad=grad) for o in self.operands])

    class TerminalABC:

        @property
        @abstractmethod
        def action(self):
            pass

        def __init__(self, shape):
            self._value = self.action(shape)

        def forward(self, grad=False):
            return self._value

    def __init__(self, ret_type):
        self.pset = gp.PrimitiveSetTyped(
            'factorization', [], ret_type
        )

    def gen_expr(self, t=None):
        return gp.genGrow(self.pset, min_=0, max_=99, type_=t)

    def instantiate(self, expr, shape, random_state=None):
        s = expr.pop(0)
        node_cls = self.pset.context[s.name]
        args = []
        for _ in range(s.arity):
            t, expr = self.instantiate(expr, shape, random_state)
            args.append(t)
        node = node_cls(shape, *args)
        return node, expr

    def add_primitive(self, name, action, in_types, ret_type):

        class Primitive(self.PrimitiveABC):
            @property
            def action(self):
                return action

        self.pset.addPrimitive(Primitive, in_types, ret_type, name=name)

    def add_terminal(self, name, action, ret_type):

        class Terminal(self.TerminalABC):
            @property
            def action(self):
                return action

        self.pset.addTerminal(Terminal, ret_type, name=name)
