#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import make_dataclass
import inspect
from numbers import Real
from typing import Optional, Tuple
from ._terminal import AbstractIndex, AbstractTensor, LiteralValue


class _ASNode:
    pass


class Primitives:

    def primitive(precedence=None):
        def make_primitive(f):
            args = inspect.getfullargspec(f).args
            p = make_dataclass(
                f.__name__,
                args,
                bases=(_ASNode,)
            )
            p.name = property(lambda self: f.__name__)
            if precedence is not None:
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
    def literal(value: LiteralValue):
        '''a literal value'''

    @primitive(precedence=0)
    def tensor(abstract: AbstractTensor):
        '''an abstract tensor'''

    @primitive(precedence=0)
    def index(item: AbstractIndex, bound: bool, kron: bool):
        '''an index; bound indices are not reduced even if they appear twice;
        kron indices lead to kronecker product between the dimensions with the
        same index.'''

    @primitive(precedence=0)
    def indices(items: Tuple[AbstractIndex]):
        '''a tuple of indices'''

    @primitive(precedence=1)
    def index_notation(tensor: _ASNode, indices: _ASNode):
        '''indexed notation for a single tensor: tensor[indices...]'''

    @primitive(precedence=2)
    def call(f: str, x: _ASNode):
        '''general function call: f(x)'''

    @primitive(precedence=3)
    def pow(base: _ASNode, exponent: _ASNode):
        '''raise to power: base**exponent'''

    @primitive(precedence=4)
    def neg(x: _ASNode):
        '''elementwise negation'''

    @primitive(precedence=None)
    def ein(
        lhs: _ASNode, rhs: _ASNode, precedence: int,
        reduction: str, pairwise: str, outidx: Optional[_ASNode]
    ):
        '''pairwise einsum-like operations between tensors'''

    @primitive(precedence=9)
    def tran(src: _ASNode, indices: _ASNode):
        '''transposition/axis reordering'''

    @classmethod
    def as_primitive(cls, raw):
        if isinstance(raw, _ASNode):
            return raw
        elif isinstance(raw, Real):
            return cls.literal(value=LiteralValue(raw))
        else:
            raise TypeError(
                f'Cannot use {raw} of type {type(raw)} in '
                f'a tensor expression.'
            )


class _AST:

    def __init__(self, data=None):
        try:  # copy-construct from another AST
            self.root = data.root
        except AttributeError:
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
