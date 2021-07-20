#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
import re
import uuid
import numbers
from . import _primitive as primitive


class LaTeXHTMLRepr(ABC):

    @abstractmethod
    def _repr_tex_(self):
        pass

    def _repr_html_(self):
        return f'''$${self._repr_tex_()}$$'''


class Symbol(LaTeXHTMLRepr):

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, string: str):
        if string is None:
            string = f'_{uuid.uuid4().hex[:8]}'

        assert string.isidentifier(), \
            f"A symbol must be a valid Python identifiers, not {repr(string)}."
        self._symbol = string

    def _repr_tex_(self):
        if len(self.symbol) > 1:
            return fr'\left[{self.symbol}\right]'
        else:
            return self.symbol


class Index(Symbol):

    def __init__(self, symbol):
        self.symbol = symbol

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return f'{type(self).__qualname__}({repr(self.symbol)})'


class Expression(LaTeXHTMLRepr):

    def __add__(self, rhs: Expression):
        return BinaryExpression(primitive.p_add, self, rhs)

    def __sub__(self, rhs: Expression):
        return BinaryExpression(primitive.p_sub, self, rhs)

    def __mul__(self, rhs: Expression):
        return BinaryExpression(primitive.p_mul, self, rhs)

    def __neg__(self):
        return UnaryExpression(primitive.p_neg, self)

    @abstractmethod
    def __repr__(self):
        pass


class LazyBase(Symbol):

    def __init__(self, *size, symbol=None, initial=None):
        for d, n in enumerate(size):
            if not (isinstance(n, numbers.Integral) and n > 0):
                raise RuntimeError(
                    "Tensor dimension must be positive integer, "
                    f"got {n} for mode {d}."
                )
        self._shape = tuple(map(int, size))
        self.symbol = symbol
        self.initial = initial

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return '{cls}({shape}, {symbol}{initial})'.format(
            cls=type(self).__qualname__,
            shape=self.shape,
            symbol=f'symbol={repr(self.symbol)}',
            initial=f', initial={repr(self.initial)}'
                    if self.initial is not None else ''
        )

    @property
    def shape(self):
        return self._shape


class LazyScalar(LazyBase, Expression):
    pass


class LazyTensor(LazyBase):
    '''A lazy tensor is a symbolic representation of a multidimensional array
    and is convenient for specifying **tensor expressions**. At construction,
    it does not allocate memory nor populate elements, but rather just record
    the shape, the method of initialization, and other related properties for a
    tensor. This is in contrast to the behavior of common linear algebra
    libraries, where multidimensional arrays are 'eager' in allocating memory
    and creating the data.

    Parameters
    ----------
    size: int...
        A sequence of integers specifying the shape of the tensor.
        Can be either a variable number of arguments or an iterable like a list
        or tuple.
    '''

    def __getitem__(self, indices):
        try:
            return IndexExpression(self, tuple(indices))
        except TypeError:
            return IndexExpression(self, (indices,))

    def _repr_tex_(self):
        return fr'''\mathbf{{{str(self.symbol)}}}'''


class IndexExpression(Expression):

    def __init__(self, tensor, indices):
        self.oper = primitive.p_idx
        self.tensor = tensor
        self.indices = indices

    def __str__(self):
        return '{tensor}[{indices}]'.format(
            tensor=str(self.tensor),
            indices=', '.join(map(str, self.indices))
        )

    def __repr__(self):
        return '{tensor}[{indices}]'.format(
            tensor=repr(self.tensor),
            indices=', '.join(map(repr, self.indices))
        )

    def _repr_tex_(self):
        idx = ''.join([i._repr_tex_() for i in self.indices])
        return fr'''{{{self.tensor._repr_tex_()}}}_{{{idx}}}'''


class UnaryExpression(Expression):

    def __init__(self, oper, operand):
        self.oper = oper
        self.operand = operand

    def __repr__(self):
        return f'{self.oper.name}({repr(self.operand)})'


class BinaryExpression(Expression):

    def __init__(self, oper, lhs, rhs):
        self.oper = oper
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        lstr = str(self.lhs)
        rstr = str(self.rhs)
        if self.lhs.oper.precedence > self.oper.precedence:
            lstr = fr'({lstr})'
        if self.rhs.oper.precedence > self.oper.precedence:
            rstr = fr'({rstr})'
        return f'{lstr} {self.oper.symbol} {rstr}'

    def __repr__(self):
        return f'{self.oper.name}({repr(self.lhs)}, {repr(self.rhs)})'

    def _repr_tex_(self):
        lstr = self.lhs._repr_tex_()
        rstr = self.rhs._repr_tex_()
        if self.lhs.oper.precedence > self.oper.precedence:
            lstr = fr'\left({lstr}\right)'
        if self.rhs.oper.precedence > self.oper.precedence:
            rstr = fr'\left({rstr}\right)'
        return f'{lstr} {self.oper.tex} {rstr}'


def index(symbol):
    return Index(symbol)


def indices(symbols):
    return [index(s) for s in re.split(r'[,\s]+', symbols)]


def tensor(*size, symbol=None, initial=None):
    if len(size) == 0:
        T = LazyScalar
    else:
        T = LazyTensor

    return T(*size, symbol=symbol, initial=initial)
