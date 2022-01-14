#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    dfs_filter,
    Compiler,
    EinsteinSpecGenerator,
    Evaluator,
    IndexnessAnalyzer,
    LeafInitializer,
    ElementwiseEvaluator,
    SlicingPropagator,
)
from funfact import active_backend as ab


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

    def __init__(self, tsrex, **extra_attributes):
        self._tsrex = (tsrex
                       | IndexnessAnalyzer()
                       | Compiler()
                       | EinsteinSpecGenerator())
        self.__dict__.update(**extra_attributes)

    @classmethod
    def from_tsrex(cls, tsrex, dtype=None, initialize=True):
        '''Construct a factorization model from a tensor expresson.

        Args:
            tsrex (TsrEx): The tensor expression.
            dtype: numerical data type, defaults to float32.
            initialize (bool):
                Whether or not to fill abstract tensors with actual data.
        '''
        if initialize:
            tsrex = tsrex | LeafInitializer(dtype)
        return cls(tsrex)

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
            list(dfs_filter(lambda n: n.name == 'tensor' and
                            n.decl.optimizable, self.tsrex.root))
        )

    @factors.setter
    def factors(self, tensors):
        for i, n in enumerate(
            dfs_filter(lambda n: n.name == 'tensor' and
                       n.decl.optimizable, self.tsrex.root)
        ):
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
            list(dfs_filter(lambda n: n.name == 'tensor', self.tsrex.root))
        )

    @property
    def tsrex(self):
        '''The underlying tensor expression.'''
        return self._tsrex

    @property
    def shape(self):
        '''The shape of the result tensor.'''
        return self.tsrex.shape

    @property
    def ndim(self):
        '''The dimensionality of the result tensor.'''
        return self.tsrex.ndim

    def penalty(self, sum_leafs: bool = True, sum_vec=None):
        '''The penalty of the result tensor.

        Args:
            sum_leafs (bool): sum the penalties over the leafs of the model.
            sum_vec (bool): sum the penalties over the vectorization dimension.
        '''

        factors = list(dfs_filter(
                lambda n: n.name == 'tensor' and n.decl.optimizable,
                self.tsrex.root)
        )
        penalties = ab.stack(
            [f.decl.prefer(f.data, sum_vec) for f in factors],
            0 if sum_vec else -1
        )
        return ab.sum(penalties, 0 if sum_vec else -1) if sum_leafs else \
            penalties

    def __call__(self):
        '''Shorthand for :py:meth:`forward`.'''
        return self.forward()

    def forward(self):
        '''Evaluate the tensor expression the result.'''
        return self.tsrex | Evaluator()

    def _get_elements(self, key):
        '''Get elements at index of tensor expression.'''

        # Generate full index list
        full_idx = []
        ellipsis_i = None
        for i in key:
            if isinstance(i, int):
                if i != -1:
                    full_idx.append(slice(i, i+1))
                else:
                    full_idx.append(slice(i, None))
            elif isinstance(i, slice):
                full_idx.append(i)
            elif i is Ellipsis:
                ellipsis_i = key.index(...)
            else:
                raise IndexError(
                    f'Unrecognized index {i} of type {type(i)}'
                )
        if ellipsis_i is not None:
            for i in range(self.ndim - len(full_idx)):
                full_idx.insert(ellipsis_i, slice(None))

        # Validate full index list
        if len(full_idx) != self.ndim:
            raise IndexError(
                f'Wrong number of indices {len(full_idx)},'
                f'expected {self.ndim}'
            )
        for i, idx in enumerate(full_idx):
            if idx.start is not None:
                if idx.start >= self.shape[i] or idx.start < -self.shape[i]:
                    raise IndexError(
                        f'index.start {idx.start} is out of bounds for '
                        f'axis {i} with size {self.shape[i]}'
                    )
            if idx.stop is not None:
                if idx.stop > self.shape[i] or idx.stop <= -self.shape[i]:
                    raise IndexError(
                        f'index.stop {idx.stop} is out of bounds for '
                        f'axis {i} with size {self.shape[i]}'
                    )

        # Evaluate model
        return self.tsrex | SlicingPropagator(full_idx) \
                          | ElementwiseEvaluator()

    def __getitem__(self, idx):
        '''Implements attribute-based access of factor tensors or output
        elements.'''
        if isinstance(idx, str):
            for n in dfs_filter(
                lambda n: n.name == 'tensor' and str(n.decl.symbol) == idx,
                self.tsrex.root
            ):
                return n.data
            raise AttributeError(f'No factor tensor named {idx}.')
        else:
            return self._get_elements(idx)

    def __setitem__(self, name, data):
        '''Implements attribute-based access of factor tensors.'''
        for n in dfs_filter(
            lambda n: n.name == 'tensor' and str(n.decl.symbol) == name,
            self.tsrex.root
        ):
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
