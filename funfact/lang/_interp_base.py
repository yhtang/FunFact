#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy


class Interpreter(ABC):
    '''An interpreter traverses an abstract syntax tree (AST) in a depth-first
    manner and perform corresponding actions for each of the node. Applying
    different interpreters to the same AST lead to different results, all of
    which conform to yet reflects the different facets of the AST. See
    definitions of the primitives in the `Primitives` class.'''

    @abstractmethod
    def __call__(self, expr):
        pass

    @abstractmethod
    def scalar(self, leaf):
        pass

    @abstractmethod
    def tensor(self, leaf):
        pass

    @abstractmethod
    def index(self, leaf):
        pass

    @abstractmethod
    def index_notation(self, tensor, *indices):
        pass

    @abstractmethod
    def call(self, tsrex, f):
        pass

    @abstractmethod
    def pow(self, base, exponent):
        pass

    @abstractmethod
    def neg(self, tsrex):
        pass

    @abstractmethod
    def mul(self, lhs, rhs):
        pass

    @abstractmethod
    def div(self, lhs, rhs):
        pass

    @abstractmethod
    def add(self, lhs, rhs):
        pass

    @abstractmethod
    def sub(self, lhs, rhs):
        pass


class FunctionalInterpreter(Interpreter):
    '''A functional interpreter traverses an AST for one pass and produces the
    final outcome without altering the AST. Intermediates are passed as return
    values between the traversing levels.'''
    def __call__(self, expr, parent=None):
        rule = getattr(self, expr.p.name)
        if expr.p.terminal:
            return rule(*expr.operands, **expr.params)
        else:
            return rule(*[self(e, expr) for e in expr.operands],
                        **expr.params)


class TranscribeInterpreter(Interpreter):
    '''A transcribe interpreter creates a modified copy of an AST while
    traversing it.'''
    def __call__(self, expr, parent=None):
        expr = copy.copy(expr)
        rule = getattr(self, expr.p.name)
        if not expr.p.terminal:
            expr.operands = tuple([self(e, expr) for e in expr.operands])
        expr.payload = rule(*expr.operands, **expr.params)
        return expr
