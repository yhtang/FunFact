#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import re
import numbers


class Identifier(ABC):

    @property
    def symbol(self):
        if self._number is not None:
            return f'{self._letter}_{self._number}'
        else:
            return self._letter

    @symbol.setter
    def symbol(self, string: str):
        m = re.fullmatch(r'([a-zA-Z]+)(?:_(\d+))?', string)
        try:
            self._letter, self._number = m.groups()
        except AttributeError:
            m = re.fullmatch(r'__(\d+)', string)
            try:
                self._letter = r'\lambda'
                self._number, = m.groups()
            except AttributeError:
                raise RuntimeError(
                    f'{repr(string)} is not a valid symbol.\n'
                    'A symbol must be alphabetic and optionally followed by '
                    'an underscore and a numeric subscript. '
                    'Examples: i, j, k_0, lhs, etc.'
                )

    @abstractmethod
    def _repr_tex_(self):
        pass

    def _repr_html_(self):
        return f'''$${self._repr_tex_()}$$'''


class AbstractIndex(Identifier):

    def __init__(self, symbol):
        self.symbol = symbol

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return f'{type(self).__qualname__}({repr(self.symbol)})'

    def _repr_tex_(self, accent=None):
        if accent is not None:
            letter = fr'{accent}{{{self._letter}}}'
        else:
            letter = self._letter
        if self._number is not None:
            return fr'{{{letter}}}_{{{self._number}}}'
        else:
            return fr'{{{letter}}}'


class AbstractTensor(Identifier):
    '''An abstract tensor is a symbolic representation of a multidimensional
    array and is convenient for specifying **tensor expressions**. At
    construction, it does not allocate memory nor populate elements, but rather
    just record the shape, the method of initializerization, and other related
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

    def __init__(self, symbol, *size, initializer=None):
        self.symbol = symbol
        for d, n in enumerate(size):
            if not (isinstance(n, numbers.Integral) and n > 0):
                raise RuntimeError(
                    "Tensor dimension must be positive integer, "
                    f"got {n} for mode {d}."
                )
        self._shape = tuple(map(int, size))
        self.initializer = initializer

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return '{cls}({symbol}, {shape}{initializer})'.format(
            cls=type(self).__qualname__,
            symbol=repr(self.symbol),
            shape=self.shape,
            initializer=f', initializer={repr(self.initializer)}'
                        if self.initializer is not None else ''
        )

    def _repr_tex_(self):
        if self._number is not None:
            return fr'\boldsymbol{{{self._letter}}}^{{({self._number})}}'
        else:
            return fr'\boldsymbol{{{self._letter}}}'
