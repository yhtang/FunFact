#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import multiprocessing
import re
import numbers
import uuid


class Symbol:

    letter: str
    number: str = None

    # to be provisioned by subclasses
    # TODO: a better apporach is to directly implement a safe dict
    _anon_registry: dict
    _anon_registry_lock: multiprocessing.Lock

    def __init__(self, identifier=None):
        if isinstance(identifier, tuple):
            self.letter, self.number = identifier
        elif isinstance(identifier, str):
            m = re.fullmatch(r'([a-zA-Z]+)(?:_(\d+))?', identifier)
            try:
                self.letter, self.number = m.groups()
            except AttributeError:
                raise RuntimeError(
                    f'{repr(identifier)} is not a valid symbol, which '
                    'must be alphabetic and optionally followed by '
                    'an underscore and a numeric subscript, such as '
                    'i, j, k_0, lhs, etc.'
                )
        elif isinstance(identifier, uuid.UUID):
            self.letter, self.number = self._make_symbol(identifier)
        else:
            raise RuntimeError(f'Cannot create symbol from {identifier}.')

    def __repr__(self):
        return f'{type(self).__qualname__}({self.letter}, {self.number})'

    def __str__(self):
        if self.number is not None:
            return f'{self.letter}_{self.number}'
        else:
            return self.letter

    @classmethod
    def _make_symbol(cls, u):
        with cls._anon_registry_lock:
            if u in cls._anon_registry:
                return cls._anon_registry[u]
            else:
                i = str(len(cls._anon_registry))
                cls._anon_registry[u] = s = ('', i)
                return s


class Identifiable(ABC):

    def __init__(self, symbol: str = None):
        self.uuid = uuid.uuid4()

    def __hash__(self):
        return self.uuid.int

    def __eq__(self, other):
        return self.uuid == other.uuid


class LaTexReprMixin(ABC):
    @abstractmethod
    def _repr_tex_(self):
        pass

    def _repr_html_(self):
        return f'''$${self._repr_tex_()}$$'''


class AbstractIndex(Identifiable, LaTexReprMixin):

    class IndexSymbol(Symbol):
        _anon_registry = {}
        _anon_registry_lock = multiprocessing.Lock()

    def __init__(self, symbol=None):
        super().__init__()
        self.symbol = self.IndexSymbol(symbol or self.uuid)

    def __str__(self):
        return str(self.symbol)

    def __repr__(self):
        return f'{type(self).__qualname__}({str(self.symbol)})'

    def _repr_tex_(self, accent=None):
        letter = self.symbol.letter or r'\#'
        number = self.symbol.number
        if accent is not None:
            letter = fr'{accent}{{{letter}}}'
        if number is not None:
            return fr'{{{letter}}}_{{{number}}}'
        else:
            return fr'{{{letter}}}'


class AbstractTensor(Identifiable, LaTexReprMixin):
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

    class TensorSymbol(Symbol):
        _anon_registry = {}
        _anon_registry_lock = multiprocessing.Lock()

    def __init__(self, *size, symbol=None, initializer=None):
        super().__init__()
        for d, n in enumerate(size):
            if not (isinstance(n, numbers.Integral) and n > 0):
                raise RuntimeError(
                    "Tensor dimension must be positive integer, "
                    f"got {n} for mode {d}."
                )
        self._shape = tuple(map(int, size))
        self.symbol = self.TensorSymbol(symbol or self.uuid)
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
        return '{cls}({shape}, {symbol}{initializer})'.format(
            cls=type(self).__qualname__,
            shape=self.shape,
            symbol=self.symbol,
            initializer=f', initializer={repr(self.initializer)}'
                        if self.initializer is not None else ''
        )

    def _repr_tex_(self):
        letter = Symbol._latex_encode(self.symbol.letter)
        number = self.symbol.number
        if number is not None:
            return fr'\boldsymbol{{{letter}}}^{{({number})}}'
        else:
            return fr'\boldsymbol{{{letter}}}'


class _SpecialTensor(Identifiable, LaTexReprMixin):
    '''A special tensor such as Kronecker delta, shifting operator, etc.

    Parameters
    ----------
    name: str
        The name of the special tensor.
    ndim: int
        Dimensionality of the special tensor.
    '''

    def __init__(self, symbol, tex, **kwargs):
        super().__init__()
        self.symbol = symbol
        self.tex = tex
        self.__dict__.update(**kwargs)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return '{cls}({symbol}, {tex})'.format(
            cls=type(self).__qualname__,
            symbol=repr(self.symbol),
            tex=self.tex
        )

    def _repr_tex_(self):
        return self.tex
