#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import torch
import numpy as np
from deap import gp
from ntf.util.iterable import flatten, map_or_call


class FactorizationPrimitiveSet:
    '''A DEAP primitive set for nonlinear tensor factorization.

    Parameters
    ----------
    ret_type: type
        Type of the overall factorization expression.
    rank_types: list
        The types of the vectors that make up one 'rank'.
    k: int
        Maximum number of ranks to be sought in the factorization.
    '''

    class PrimitiveBase(ABC):

        @property
        @abstractmethod
        def action(self):
            pass

        def __init__(self, shape, *operands):
            self.operands = tuple(operands)

        def forward(self, grad=False):
            return self.action(*[o.forward(grad=grad) for o in self.operands])

    class TerminalBase(ABC):

        @property
        @abstractmethod
        def action(self):
            pass

        def __init__(self, shape):
            self._value = self.action(shape)

        def forward(self, grad=False):
            return self._value

    def __init__(self, ret_type: type, rank_types: list, k=10):
        self.ret_type = ret_type
        self.rank_types = rank_types
        self.k = k
        self.pset = gp.PrimitiveSetTyped(
            'factorization', rank_types * k, ret_type
        )

    def gen_expr(self, max_depth: int, p=None):
        '''Propose a candidate nonlinear factorization expression.

        Parameters
        ----------
        max_depth: int
            Maximum depth (number of layers) of the expression.
        p: dict or callable
            A lookup table of the relative frequencies of the primitives in the
            generated expression.

        Returns
        -------
        expr: list
            A factorization in the form of a prefix expression.
        '''
        return self._gen_expr(
            self.ret_type,
            p if p is not None else lambda _: 1.0,
            max_depth
        )

    def _gen_expr(self, t=None, p=None, d=0):
        if d <= 0:  # try to terminate ASAP
            try:
                choice = np.random.choice(self.pset.terminals[t], 1).item()
            except ValueError:
                choice = np.random.choice(self.pset.primitives[t], 1).item()
        else:  # normal growth
            candidates = self.pset.primitives[t] + self.pset.terminals[t]
            prob = np.fromiter(map_or_call(candidates, p), dtype=np.float)
            choice = np.random.choice(
                candidates, 1, p=prob / prob.sum()
            ).item()

        if isinstance(choice, gp.Terminal):
            return [choice]
        else:
            return [choice, *flatten([self._gen_expr(a, p=p, d=d-1)
                                      for a in choice.args])]

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

        class Primitive(self.PrimitiveBase):
            @property
            def action(self):
                return action

        self.pset.addPrimitive(Primitive, in_types, ret_type, name=name)

    def add_terminal(self, name, action, ret_type):

        class Terminal(self.TerminalBase):
            @property
            def action(self):
                return action

        self.pset.addTerminal(Terminal, ret_type, name=name)
