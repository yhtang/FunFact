#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
# from typing import NamedTuple, Any


# class Evaluated(NamedTuple):
#     expr: Any
#     value: Any


def no_op(f):
    '''A no-op rule does nothing.'''
    def do_nothing(*args, **kwargs):
        pass

    return do_nothing


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
    def scalar(self):
        pass

    @abstractmethod
    def tensor(self):
        pass

    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def index_notation(self):
        pass

    @abstractmethod
    def call(self):
        pass

    @abstractmethod
    def pow(self):
        pass

    @abstractmethod
    def neg(self):
        pass

    @abstractmethod
    def mul(self):
        pass

    @abstractmethod
    def div(self):
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def sub(self):
        pass


class FunctionalInterpreter(Interpreter):
    '''A functional interpreter traverses an AST for one pass and produces the
    final outcome without altering the AST. Intermediates are passed as return
    values between the traversing levels.'''
    def __call__(self, expr, parent=None):
        return getattr(self, expr.p.name)(
            *[self(operand, expr) for operand in expr.operands], **expr.params
        )


class ImperativeInterpreter(Interpreter):
    '''An imperative interpreter traverses an AST and then either modifies it or
    creates a new AST as a result of the evaluation.'''
    def __call__(self, expr):
        # TBD
        pass
