#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import uuid
import numbers
from . import _primitive as P


class Symbol:

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


class Index(Symbol):

    def __init__(self, symbol):
        self.symbol = symbol


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

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, indices):
        try:
            return P.Expression(P.p_idx, self, *indices)
        except TypeError:
            return P.Expression(P.p_idx, self, indices)


def index(symbol):
    return Index(symbol)


def indices(symbols):
    return [index(s) for s in re.split(r'[,\s]+', symbols)]


def tensor(*size, symbol=None, initial=None):
    return AbstractTensor(*size, symbol=symbol, initial=initial)
