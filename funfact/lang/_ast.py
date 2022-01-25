#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import make_dataclass
import inspect
from numbers import Real
from typing import Optional, Union, Tuple
from ._terminal import AbstractIndex, AbstractTensor, LiteralValue


class _ASNode:
    pass


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


class Primitives:

    @primitive(precedence=1)
    def abstract_index_notation(tensor: _ASNode, indices: _ASNode):
        '''indexing a raw tensor or tensor expression: tensor[indices...].
        To be interpreted either as tensor indexing or index renaming depending
        on the type of the addressee.'''

    @primitive(precedence=None)
    def abstract_binary(
        lhs: _ASNode, rhs: _ASNode, precedence: int, operator: str
    ):
        '''generic binary operations, to be interpreted as  matmul, kron,
        elementwise, or Einstein based on index/indexless status and the
        operator being used.'''

    @primitive(precedence=0)
    def literal(value: LiteralValue):
        '''a literal value'''

    @primitive(precedence=0)
    def tensor(decl: AbstractTensor):
        '''an abstract tensor'''

    @primitive(precedence=0)
    def index(item: AbstractIndex, bound: bool, kron: bool):
        '''an index; bound indices are not reduced even if they appear twice;
        kron indices lead to kronecker product between the dimensions with the
        same index.'''

    @primitive(precedence=0)
    def indices(items: Tuple[index]):
        '''a tuple of indices'''

    @primitive(precedence=1)
    def indexed_tensor(tensor: _ASNode, indices: _ASNode):
        '''indexing a raw tensor or indexless expression: expr[indices...]'''

    @primitive(precedence=2)
    def call(f: str, x: _ASNode):
        '''general function call: f(x)'''

    @primitive(precedence=4)
    def neg(x: _ASNode):
        '''elementwise negation'''

    @primitive(precedence=None)
    def elem(
        lhs: _ASNode, rhs: _ASNode, precedence: int, operator: str
    ):
        '''indexless elementwise operations between tensors'''

    @primitive(precedence=None)
    def ein(
        lhs: _ASNode, rhs: _ASNode, precedence: int,
        reduction: str, pairwise: str, outidx: Optional[_ASNode]
    ):
        '''pairwise einsum-like operations between tensors'''

    @primitive(precedence=9)
    def tran(src: _ASNode, indices: _ASNode):
        '''transpose the tensor by permuting the axes.'''

    @primitive(precedence=9)
    def abstract_dest(src: _ASNode, indices: indices):
        '''generic destination index designation; maybe translated either
        into transposition/axis permutation, or specify the output indices
        of Einstein operations'''

    Tensorial = Union[
        indexed_tensor, call, neg, ein
    ]
    Numeric = Union[Tensorial, Real]


class _AST:

    def __init__(self, root=None):
        self.root = root

    @property
    def root(self):
        '''Root node of the tensor expression'''
        return self._root

    @root.setter
    def root(self, r):
        self._root = r
