#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import re
import uuid
import numbers
from ._expr import Expr


class Symbol(ABC):

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, string: str):
        assert string.isidentifier(), \
            f"A symbol must be a valid Python identifiers, not {repr(string)}."
        self._symbol = string

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
        # return f'''{str(self.symbol)}'''
        if len(self.symbol) > 1:
            return fr'\left[{self.symbol}\right]'
        else:
            return self.symbol


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

    def __getitem__(self, indices):
        try:
            return Expr('idx', self, *indices)
        except TypeError:
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
        return fr'''\mathbf{{{str(self.symbol)}}}'''


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
        symbol = f'_{uuid.uuid4().hex[:8]}'
        size = spec

    return AbstractTensor(symbol, *size, initial=initial)
