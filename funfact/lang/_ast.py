#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import make_dataclass, field
import inspect
from numbers import Real
from typing import Iterable, Union, Any
from ._tensor import AbstractIndex, AbstractTensor


class _ASNode:
    pass


class Primitives:

    def primitive(precedence):
        def make_primitive(f):
            p = make_dataclass(
                f.__name__,
                inspect.getfullargspec(f).args + [('payload', Any, field(default=None))],
                bases=(_ASNode,)
            )
            p.name = property(lambda self: f.__name__)
            p.precedence = property(lambda self: precedence)
            return p
        return make_primitive

    @primitive(precedence=0)
    def scalar(value):
        '''an index'''

    @primitive(precedence=0)
    def index(value: AbstractIndex):
        '''an abstract tensor'''

    @primitive(precedence=0)
    def tensor(value: AbstractTensor):
        '''a scalar number'''

    @primitive(precedence=1)
    def index_notation(
        tensor: _ASNode,
        indices: Union[AbstractIndex, Iterable[AbstractIndex]]
    ):
        '''indexed notation for a single tensor'''

    @primitive(precedence=2)
    def call(f: str, x: _ASNode):
        '''nonlinear function call'''

    @primitive(precedence=3)
    def pow(base: _ASNode, exponent: _ASNode):
        '''raise to power'''

    @primitive(precedence=4)
    def neg(x: _ASNode):
        '''elementwise negation'''

    @primitive(precedence=5)
    def mul(lhs: _ASNode, rhs: _ASNode):
        '''elementwise multiplication, Hadamard product'''

    @primitive(precedence=5)
    def div(lhs: _ASNode, rhs: _ASNode):
        '''elementwise division'''

    @primitive(precedence=6)
    def add(lhs: _ASNode, rhs: _ASNode):
        '''elementwise addition'''

    @primitive(precedence=6)
    def sub(lhs: _ASNode, rhs: _ASNode):
        '''elementwise subtraction'''

    @classmethod
    def as_primitive(cls, value):
        if isinstance(value, _ASNode):
            return value
        elif isinstance(value, Real):
            return cls.scalar(value=value)
        else:
            raise TypeError(
                f'Cannot use {value} of type {type(value)} in'
                'a tensor expression.'
            )


class _AST:

    def __init__(self, data=None):
        if isinstance(data, type(self)):
            self.root = data.root
        else:
            try:
                self.root = Primitives.as_primitive(data)
            except TypeError:
                raise RuntimeError(
                    f'Invalid arguments to create an AST: data = {data}.'
                )

    def astree(self, t):
        return type(self)(t)

    def asnode(self, t):
        return self.astree(t).root
