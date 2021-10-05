#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numbers
import re
import typing
from ._expr import Expr


class Symbol(ABC):

    @property
    def symbol(self):
        if self._number is not None:
            return f'{self._letter}_{self._number}'
        else:
            return self._letter

    @symbol.setter
    def symbol(self, string: str):
        m = re.fullmatch('([a-zA-Z]+)(?:_(\d+))?', string)
        if m is None:
            raise RuntimeError(
                f'{repr(string)} is not a valid symbol.\n'
                'A symbol must be alphabetic and optionally followed by an '
                'underscore and a numeric subscript. '
                'Examples: i, j, k_0, lhs, etc.'
            )
        self._letter, self._number = m.groups()
        if self._letter == 'Anonymous':
            self._letter = r'\varphi'

    @abstractmethod
    def _repr_tex_(self):
        pass

    def _repr_html_(self):
        return f'''$${self._repr_tex_()}$$'''


class Index(Symbol):

    def __init__(self, symbol):
        self.symbol = symbol

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return f'{type(self).__qualname__}({repr(self.symbol)})'

    def _repr_tex_(self):
        if self._number is not None:
            return fr'{{{self._letter}}}_{{{self._number}}}'
        else:
            return fr'{{{self._letter}}}'


class AbstractTensor(Symbol):
    '''An abstract tensor is a symbolic representation of a multidimensional
    array and is convenient for specifying **tensor expressions**. At
    construction, it does not allocate memory nor populate elements, but rather
    just record the shape, the method of initialization, and other related
    properties for a tensor. This is in contrast to the behavior of common
    linear algebra libraries, where multidimensional arrays are 'eager' in
    allocating memory and creating the data.

    Parameters
    ----------
    size: int...
        A sequence of integers specifying the shape of the tensor.
        Can be either a variable number of arguments or an iterable like a list
        or tuple.
    '''

    n_nameless = 0

    def __init__(self, symbol, *size, initial=None):
        self.symbol = symbol
        for d, n in enumerate(size):
            if not (isinstance(n, numbers.Integral) and n > 0):
                raise RuntimeError(
                    "Tensor dimension must be positive integer, "
                    f"got {n} for mode {d}."
                )
        self._shape = tuple(map(int, size))
        self.initial = initial

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __getitem__(self, indices):
        if isinstance(indices, typing.Iterable):
            assert len(indices) == self.ndim,\
                f"Indices {indices} does not match the rank of tensor {self}."
            return Expr('idx', self, *indices)
        else:
            assert 1 == self.ndim,\
                f"Indices {indices} does not match the rank of tensor {self}."
            return Expr('idx', self, indices)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return '{cls}({symbol}, {shape}{initial})'.format(
            cls=type(self).__qualname__,
            symbol=repr(self.symbol),
            shape=self.shape,
            initial=f', initial={repr(self.initial)}'
                    if self.initial is not None else ''
        )

    def _repr_tex_(self):
        if self._number is not None:
            return fr'\boldsymbol{{{self._letter}}}^{{({self._number})}}'
        else:
            return fr'\boldsymbol{{{self._letter}}}'


def index(symbol):
    return Index(symbol)


def indices(symbols):
    return [index(s) for s in re.split(r'[,\s]+', symbols)]


def tensor(*spec, initial=None):
    '''Construct an abstract tensor using `spec`.

    Parameters
    ----------
    spec:
        Formats supported:

        * symbol, size...
        * size...

    initial:
        Initialization distribution
    '''
    if isinstance(spec[0], str):
        symbol, *size = spec
    else:
        symbol = f'Anonymous_{AbstractTensor.n_nameless}'
        AbstractTensor.n_nameless += 1
        size = spec

    return AbstractTensor(symbol, *size, initial=initial)
