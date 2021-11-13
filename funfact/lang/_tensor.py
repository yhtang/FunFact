#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass
import multiprocessing
import re
import numbers
import uuid


@dataclass(init=False)
class Symbol:

    letter: str
    number: str = None

    def __init__(self, s):
        if isinstance(s, tuple):
            self.letter, self.number = s
        else:
            m = re.fullmatch(r'([a-zA-Z]+)(?:_(\d+))?', s)
            try:
                self.letter, self.number = m.groups()
            except AttributeError:
                raise RuntimeError(
                    f'{repr(s)} is not a valid symbol, which '
                    'must be alphabetic and optionally followed by '
                    'an underscore and a numeric subscript, such as '
                    'i, j, k_0, lhs, etc.'
                )

    def __str__(self):
        if self.number is not None:
            return f'{self.letter}_{self.number}'
        else:
            return self.letter


class Identifiable(ABC):

    @staticmethod
    def _latex_encode(s):
        if s == '#':
            return r'\#'
        elif s == u'λ':
            return r'\lambda'
        else:
            return s

    def __init__(self, symbol: str = None):
        self.uuid = uuid.uuid4()
        if symbol is not None:
            self.symbol = Symbol(symbol)
        else:
            self.symbol = self._make_symbol(self.uuid)

    def __hash__(self):
        return self.uuid.int

    @abstractmethod
    def _repr_tex_(self):
        pass

    def _repr_html_(self):
        return f'''$${self._repr_tex_()}$$'''

    def __eq__(self, other):
        return self.uuid == other.uuid

    @classmethod
    def _make_symbol(cls, u):
        with cls._lock:
            if u in cls._anon_regitry:
                return cls._anon_regitry[u]
            else:
                i = str(len(cls._anon_regitry))
                cls._anon_regitry[u] = s = Symbol((cls._natural_letter, i))
                return s


class AbstractIndex(Identifiable):

    _anon_regitry = {}
    _lock = multiprocessing.Lock()
    _natural_letter = '#'
    '''the default letter to use for anonymous indices.'''

    def __init__(self, symbol=None):
        super().__init__(symbol)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return f'{type(self).__qualname__}({repr(self.symbol)})'

    def _repr_tex_(self, accent=None):
        letter = self._latex_encode(self.symbol.letter)
        number = self.symbol.number
        if accent is not None:
            letter = fr'{accent}{{{letter}}}'
        if number is not None:
            return fr'{{{letter}}}_{{{number}}}'
        else:
            return fr'{{{letter}}}'


class AbstractTensor(Identifiable):
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

    _anon_regitry = {}
    _lock = multiprocessing.Lock()
    _natural_letter = u'λ'
    '''the default letter to use for anonymous tensors.'''

    def __init__(self, *size, symbol=None, initializer=None):
        super().__init__(symbol)
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
        letter = self._latex_encode(self.symbol.letter)
        number = self.symbol.number
        if number is not None:
            return fr'\boldsymbol{{{letter}}}^{{({number})}}'
        else:
            return fr'\boldsymbol{{{letter}}}'
