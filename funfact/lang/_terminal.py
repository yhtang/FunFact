#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import multiprocessing
import re
import numbers
import uuid
from funfact.initializers import stack
from funfact.conditions import vmap


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
        if self.number is not None:
            self.number = str(self.number)

    def __repr__(self):
        return '{typename}({identifier})'.format(
            typename=type(self).__qualname__,
            identifier=repr((self.letter, self.number))
        )

    def __str__(self):
        letter = self.letter or ''
        if self.number is not None:
            return f'{letter}_{self.number}'
        else:
            return letter

    @classmethod
    def _make_symbol(cls, u):
        with cls._anon_registry_lock:
            if u in cls._anon_registry:
                return cls._anon_registry[u]
            else:
                i = str(len(cls._anon_registry))
                cls._anon_registry[u] = s = (None, i)
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


class LiteralValue(Identifiable, LaTexReprMixin):

    def __init__(self, raw, latex=None):
        super().__init__()
        self.raw = raw
        self.latex = latex

    def __str__(self):
        return str(self.raw)

    def __repr__(self):
        return '{typename}({raw}, {latex})'.format(
            typename=type(self).__qualname__,
            raw=repr(self.raw),
            latex=repr(self.latex),
        )

    def _repr_tex_(self):
        return self.latex or str(self)


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
    size: int...:
        A sequence of integers specifying the shape of the tensor.
        Can be either a variable number of arguments or an iterable like a list
        or tuple.

    symbol (str):
        An alphanumeric symbol representing the abstract tensor

    initializer (callable):
        Initialization distribution

    optimizable (boolean):
        True/False flag indicating of the abstract tensor can be optimized.

    prefer (callable):
        Condition evaluated as penalty on abstract tensor.
    '''

    class TensorSymbol(Symbol):
        _anon_registry = {}
        _anon_registry_lock = multiprocessing.Lock()

    def __init__(self, *size, symbol=None, initializer=None, optimizable=True,
                 prefer=None):
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
        self.optimizable = optimizable
        self.prefer = prefer

    def vectorize(self, n, append):
        '''Extend dimensionality by one.'''
        shape = (*self._shape, n) if append else (n, *self._shape)
        if self.initializer is None:
            initializer = self.initializer
        elif callable(self.initializer):
            initializer = stack(self.initializer, append)
        else:
            initializer = self.initializer[..., None] if append else \
                          self.initializer[None, ...]
        prefer = vmap(self.prefer, append) if self.prefer else None
        return type(self)(
            *shape, initializer=initializer, optimizable=self.optimizable,
            prefer=prefer
        )

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
        letter = self.symbol.letter or r'\lambda'
        number = self.symbol.number
        if number is not None:
            return fr'\boldsymbol{{{letter}}}^{{({number})}}'
        else:
            return fr'\boldsymbol{{{letter}}}'
