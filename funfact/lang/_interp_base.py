#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
from numbers import Real
from typing import Iterable, Union, Any
from ._ast import _ASNode, Primitives as P
from ._tensor import AbstractIndex, AbstractTensor


'''An interpreter traverses an abstract syntax tree (AST) in a depth-first
manner and perform corresponding actions for each of the node. Applying
different interpreters to the same AST lead to different results, all of
which conform to yet reflects the different facets of the AST.

**See definitions of the primitives in the `Primitives` class.**'''


def _deep_apply(f, value, *args, **kwargs):
    if isinstance(value, (list, tuple)):
        return [_deep_apply(f, elem, *args, **kwargs) for elem in value]
    elif isinstance(value, _ASNode):
        return f(value, *args, **kwargs)
    else:
        return value

def _binary_deep_apply(f, left, right):
    if isinstance(left, (list, tuple)):
        return [_binary_deep_apply(f, *elem) for elem in zip(left, right)]
    elif isinstance(left, _ASNode):
        return f(left,right)
    else:
        return left


class ROOFInterpreter(ABC):
    '''A ROOF (Read-Only On-the-Fly) interpreter traverses an AST for one pass
    and produces the final outcome without altering the AST. Intermediates are
    passed as return values between the traversing levels. Its primitive rules
    may still accept a 'payload' argument, which could be potentially produced
    by another transcribe interpreter.'''

    @abstractmethod
    def scalar(self, value: Real, payload: Any):
        pass

    @abstractmethod
    def tensor(self, value: AbstractTensor, payload: Any):
        pass

    @abstractmethod
    def index(self, value: AbstractIndex, payload: Any):
        pass

    @abstractmethod
    def index_notation(
        self, tensor: Any, indices: Iterable[Any], payload: Any
    ):
        pass

    @abstractmethod
    def call(self, f: str, x: Any, payload: Any):
        pass

    @abstractmethod
    def pow(self, base: Any, exponent: Any, payload: Any):
        pass

    @abstractmethod
    def neg(self, x: Any, payload: Any):
        pass

    @abstractmethod
    def mul(self, lhs: Any, rhs: Any, payload: Any):
        pass

    @abstractmethod
    def div(self, lhs: Any, rhs: Any, payload: Any):
        pass

    @abstractmethod
    def add(self, lhs: Any, rhs: Any, payload: Any):
        pass

    @abstractmethod
    def sub(self, lhs: Any, rhs: Any, payload: Any):
        pass

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        operands = {
            name: _deep_apply(self, value, node)
            for name, value in node.__dict__.items()
        }
        rule = getattr(self, node.name)
        return rule(**operands)


class TranscribeInterpreter(ABC):
    '''A transcribe interpreter creates a modified copy of an AST while
    traversing it.'''
    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.mul, P.div, P.add, P.sub
    ]
    Numeric = Union[Tensorial, Real]

    @abstractmethod
    def scalar(self, value: Real, payload: Any):
        pass

    @abstractmethod
    def tensor(self, value: AbstractTensor, payload: Any):
        pass

    @abstractmethod
    def index(self, value: AbstractIndex, payload: Any):
        pass

    @abstractmethod
    def index_notation(
        self, tensor: P.tensor, indices: Iterable[P.index], payload: Any
    ):
        pass

    @abstractmethod
    def call(self, f: str, x: Tensorial, payload: Any):
        pass

    @abstractmethod
    def pow(self, base: Numeric, exponent: Numeric, payload: Any):
        pass

    @abstractmethod
    def neg(self, x: Numeric, payload: Any):
        pass

    @abstractmethod
    def mul(self, lhs: Numeric, rhs: Numeric, payload: Any):
        pass

    @abstractmethod
    def div(self, lhs: Numeric, rhs: Numeric, payload: Any):
        pass

    @abstractmethod
    def add(self, lhs: Numeric, rhs: Numeric, payload: Any):
        pass

    @abstractmethod
    def sub(self, lhs: Numeric, rhs: Numeric, payload: Any):
        pass

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        node = copy.copy(node)
        node.__dict__.update(**{
            name: _deep_apply(self, value, node)
            for name, value in node.__dict__.items()
        })
        rule = getattr(self, node.name)
        node.payload = rule(**node.__dict__)
        return node

class MergeInterpreter:
    '''The merge interpreter merges the payload of two identical ASTs 
    into another one while traversing both of them.'''
    def __call__(self, left: _ASNode, right: _ASNode):
        left = copy.copy(left)
        for name in left.__dict__.keys():
            left.__dict__[name] = _binary_deep_apply(self, \
                             getattr(left, name), getattr(right, name))
        left.payload = (left.payload, right.payload)
        return left