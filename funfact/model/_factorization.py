#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Integral
from funfact import active_backend as ab
from funfact.lang.interpreter import (
    dfs_filter,
    TypeDeducer,
    EinopCompiler,
    Evaluator,
    IndexnessAnalyzer,
    LeafInitializer,
    ElementwiseEvaluator,
    SlicingPropagator,
)
from funfact.util.iterable import unique
from funfact.vectorization import vectorize


class Factorization:
    '''A factorization model realizes a tensor expression to approximate a
    target tensor.

    !!! note
        Please use one of the `from_*` class methods to construct a
        factorization model. The `__init__` method is **NOT** recommended for
        direct usage.

    Args:
        tsrex (TsrEx): A tensor expression.
        extra_attributes (kwargs): extra attributes to be stored in verbatim.

    Examples:
        >>> import funfact as ff
        >>> a = ff.tensor('a', 2, 3)
        >>> b = ff.tensor('b', 3, 4)
        >>> i, j, k = ff.indices(3)
        >>> ff.Factorization.from_tsrex(a[i, j] * b[j, k])
        <funfact.model._factorization.Factorization object at 0x7f5838105ee0>
    '''

    def __init__(self, tsrex, _secret=None, **extra_attributes):
        if _secret != '50A-2117':
            raise RuntimeError(
                'Please use one of the `from_*` methods to create a '
                'factorization from a tensor expression'
            )
        self.tsrex = tsrex
        self.__dict__.update(**extra_attributes)

    @classmethod
    def from_tsrex(
        cls, tsrex, dtype=None, vec_size=None, vec_axis=0, initialize=True
    ):
        '''Construct a factorization model from a tensor expresson.

        Args:
            tsrex (TsrEx): The tensor expression.
            dtype: numerical data type, defaults to float32.
            vec_size (int):
                Whether to vectorize the tensor expression with parallel
                instances.
            vec_axis (0 or -1): The position of the vectorization dimension.
            initialize (bool):
                Whether or not to fill abstract tensors with actual data.
        '''
        if vec_size:
            tsrex = vectorize(
                tsrex, vec_size, append=True if vec_axis == -1 else False
            )
        tsrex = tsrex | IndexnessAnalyzer() | TypeDeducer() | EinopCompiler()
        if initialize:
            tsrex = tsrex | LeafInitializer(dtype)
        return cls(tsrex, _secret='50A-2117')

    @classmethod
    def _from_jax_flatten(cls, tsrex, factors):
        '''
        '''
        tsrex = tsrex | IndexnessAnalyzer()
        fac = cls(tsrex, _secret='50A-2117')
        fac.factors = factors
        return fac

    @property
    def factors(self):
        '''A flattened list of optimizable factors in the model.

        Examples:
            >>> import funfact as ff
            >>> a = ff.tensor('a', 2, 3, optimizable=False)
            >>> b = ff.tensor('b', 3, 4)
            >>> i, j, k = ff.indices(3)
            >>> fac = ff.Factorization.from_tsrex(
            ...     a[i, j] * b[j, k],
            ...     initialize=True
            ... )
            >>> fac.factors
            <'data' field of tensor b>
            >>> fac.factors[0]
            DeviceArray([[ 0.5920733 ,  0.17746426, -1.8907379 , -0.10324025],
                         [ 0.05991533,  2.5538554 ,  0.05718338,  0.8887682 ],
                         [ 0.54816544,  2.3392196 ,  1.1973379 ,  0.04005199]],
                          dtype=float32)
        '''
        return self._NodeView(
            'data',
            list(unique(dfs_filter(
                lambda n: n.name in ['tensor', 'parametrized_tensor'] and
                n.decl.optimizable,
                self.tsrex.root
            )))
        )

    @factors.setter
    def factors(self, tensors):
        for i, n in enumerate(unique(
            dfs_filter(lambda n: n.name in ['tensor', 'parametrized_tensor']
                       and n.decl.optimizable, self.tsrex.root)
        )):
            n.data = tensors[i]

    @property
    def all_factors(self):
        '''A flattened list of all factors in the model.

        Examples:
            >>> import funfact as ff
            >>> a = ff.tensor('a', 2, 3, optimizable=False)
            >>> b = ff.tensor('b', 3, 4)
            >>> i, j, k = ff.indices(3)
            >>> fac = ff.Factorization.from_tsrex(
            ...     a[i, j] * b[j, k],
            ...     initialize=True
            ... )
            >>> fac.all_factors
            <'data' fields of tensors a, b>
            >>> fac.all_factors[0]
            DeviceArray([[[ 0.2509914 ],
                          [-0.5063717 ],
                          [-1.0069973 ]],
                         [[ 1.1088423 ],
                          [ 0.31595513],
                          [-0.11492359]]], dtype=float32)
        '''
        return self._NodeView(
            'data',
            list(unique(dfs_filter(
                lambda n: n.name == 'tensor', self.tsrex.root
            )))
        )

    @property
    def tsrex(self):
        '''The underlying tensor expression.'''
        return self._tsrex

    @tsrex.setter
    def tsrex(self, tsrex):
        '''Setting the underlying tensor expression.'''
        self._tsrex = tsrex

    @property
    def shape(self):
        '''The shape of the result tensor.'''
        return self.tsrex.shape

    @property
    def ndim(self):
        '''The dimensionality of the result tensor.'''
        return self.tsrex.ndim

    def penalty(self, sum_leafs: bool = True, sum_vec=False):
        '''The penalty of the result tensor.

        Args:
            sum_leafs (bool): sum the penalties over the leafs of the model.
            sum_vec (bool): sum the penalties over the vectorization dimension.
        '''

        factors = list(unique(dfs_filter(
                lambda n: n.name == 'tensor' and n.decl.optimizable,
                self.tsrex.root
        )))
        penalties = ab.stack(
            [f.decl.prefer(f.data, sum_vec) for f in factors],
            0 if sum_vec else -1
        )
        if sum_leafs:
            return ab.sum(penalties, 0 if sum_vec else -1)
        else:
            return penalties

    def __call__(self):
        '''Shorthand for :py:meth:`forward`.'''
        return self.forward()

    def forward(self):
        '''Evaluate the tensor expression the result.'''
        return self.tsrex | Evaluator()

    @staticmethod
    def _as_slice(i, axis):
        if isinstance(axis, slice):
            return axis
        elif isinstance(axis, Integral):
            if axis != -1:
                return slice(axis, axis + 1)
            else:
                return slice(axis, None)
        elif hasattr(axis, '__iter__'):
            return tuple(axis)
        elif axis is Ellipsis:
            return None
        else:
            raise RuntimeError(
                f'Invalid index for axis {i}: {axis}'
            )

    def _get_elements(self, key):
        '''Get elements at index of tensor expression.'''
        # Generate full index list
        indices = tuple([self._as_slice(i, axis) for i, axis in
                        enumerate(key)])
        try:
            i = key.index(Ellipsis)
        except ValueError:
            pass
        else:
            indices = tuple([
                *indices[:i],
                *[slice(None)] * (self.ndim - len(indices) + 1),
                *indices[i + 1:]
            ])

        # Validate full index list
        if len(indices) != self.ndim:
            raise IndexError(
                f'Wrong number of indices: expected {self.ndim}, '
                f'got {len(indices)}.'
            )

        # Evaluate model
        return self.tsrex | SlicingPropagator(indices) \
                          | ElementwiseEvaluator()

    def __getitem__(self, idx):
        '''Implements attribute-based access of factor tensors or output
        elements.'''
        if isinstance(idx, str):
            for n in unique(dfs_filter(
                lambda n: n.name == 'tensor' and str(n.decl.symbol) == idx,
                self.tsrex.root
            )):
                return n.data
            raise AttributeError(f'No factor tensor named {idx}.')
        else:
            return self._get_elements(idx)

    def __setitem__(self, name, data):
        '''Implements attribute-based access of factor tensors.'''
        for n in unique(dfs_filter(
            lambda n: n.name == 'tensor' and str(n.decl.symbol) == name,
            self.tsrex.root
        )):
            return setattr(n, 'data', data)
        raise AttributeError(f'No factor tensor named {name}.')

    class _NodeView:
        def __init__(self, attribute: str, nodes):
            self.attribute = attribute
            self.nodes = nodes

        def __repr__(self):
            return '<{attr} field{pl} of tensor{pl} {tensors}>'.format(
                attr=repr(self.attribute),
                tensors=', '.join([str(n.decl) for n in self.nodes]),
                pl='s' if len(self.nodes) > 1 else ''
            )

        def __getitem__(self, i):
            return getattr(self.nodes[i], self.attribute)

        def __setitem__(self, i, value):
            setattr(self.nodes[i], self.attribute, value)

        def __iter__(self):
            for n in self.nodes:
                yield getattr(n, self.attribute)

        def __len__(self):
            return len(self.nodes)
