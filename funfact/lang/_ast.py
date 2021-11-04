#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import make_dataclass
import inspect
from numbers import Real
from typing import Iterable, Union
from ._tensor import AbstractIndex, AbstractTensor


class _ASNode:
    pass


class Primitives:

    def primitive(precedence):
        def make_primitive(f):
            args = inspect.getfullargspec(f).args
            p = make_dataclass(
                f.__name__,
                args,
                bases=(_ASNode,)
            )
            p.name = property(lambda self: f.__name__)
            p.precedence = property(lambda self: precedence)
            p.fields = property(lambda self: self.__dict__)
            p.fields_fixed = property(lambda self: {
                k: v for k, v in self.__dict__.items() if k in args
            })
            p.fields_payload = property(lambda self: {
                k: v for k, v in self.__dict__.items() if k not in args
            })
            return p
        return make_primitive

    @primitive(precedence=0)
    def scalar(value):
        '''a scalar number'''

    @primitive(precedence=0)
    def index(value: AbstractIndex):
        '''an index'''

    @primitive(precedence=0)
    def tensor(value: AbstractTensor):
        '''an abstract tensor'''

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

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, r):
        self._root = r

    @classmethod
    def _as_tree(cls, t):
        return cls(t)

    @classmethod
    def _as_node(cls, t):
        return cls._as_tree(t).root
