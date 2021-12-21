#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    dfs_filter,
    EinsteinSpecGenerator,
    Evaluator,
    LeafInitializer,
    NoOp,
    IndexPropagator,
    ElementwiseEvaluator,
    SlicingPropagator,
    ShapeAnalyzer,
    Vectorizer
)


class Factorization:
    '''A factorization model realizes a tensor expression to approximate a
    target tensor.

    !!! note
        Please use one of the `from_*` class methods to construct a
        factorization model. The `__init__` method is **NOT** recommended for
        direct usage.

    Args:
        tsrex (TsrEx): A tensor expression.
        nvec (int): The number of parallel instances contained in the model.
        extra_attributes (kwargs): extra attributes to be stored in verbatim.

    Examples:
        >>> import funfact as ff
        >>> a = ff.tensor('a', 2, 3)
        >>> b = ff.tensor('b', 3, 4)
        >>> i, j, k = ff.indices(3)
        >>> ff.Factorization.from_tsrex(a[i, j] * b[j, k])
        <funfact.model._factorization.Factorization object at 0x7f5838105ee0>
    '''

    def __init__(self, tsrex, nvec, **extra_attributes):
        self._tsrex = (tsrex
                       | IndexPropagator()
                       | EinsteinSpecGenerator()
                       | ShapeAnalyzer())
        self._nvec = nvec
        self.__dict__.update(**extra_attributes)

    @classmethod
    def from_tsrex(cls, tsrex, initialize=True, nvec=1):
        '''Construct a factorization model from a tensor expresson.

        Args:
            tsrex (TsrEx): The tensor expression.
            initialize (bool):
                Whether or not to fill abstract tensors with actual data.
            nvec (int > 0):
                Number of parallel random instances to create.
        '''
        return cls(
            tsrex=(tsrex
                   | IndexPropagator()
                   | Vectorizer(nvec)
                   | (LeafInitializer() if initialize else NoOp())),
            nvec=nvec,
            _tsrex_original=tsrex,
        )

    @property
    def factors(self):
        '''A flattened list of optimizable factors in the model.

        Examples:
            >>> import funfact as ff
            >>> a = ff.tensor('a', 2, 3)
            >>> b = ff.tensor('b', 3, 4)
            >>> i, j, k = ff.indices(3)
            >>> fac = ff.Factorization.from_tsrex(
            ...     a[i, j] * b[j, k],
            ...     initialize=True
            ... )
            >>> fac.factors
            <'data' fields of tensors a, b>
            >>> fac.factors[0]
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

    @factors.setter
    def factors(self, tensors):
        for i, n in enumerate(
            dfs_filter(lambda n: n.name == 'tensor', self.tsrex.root)
        ):
            n.data = tensors[i]

    @property
    def tsrex(self):
        '''The underlying tensor expression.'''
        return self._tsrex

    @property
    def nvec(self):
        return self._nvec

    @property
    def shape(self):
        '''The shape of the result tensor.'''
        return self.tsrex.shape

    @property
    def ndim(self):
        '''The dimensionality of the result tensor.'''
        return self.tsrex.ndim

    def view(self, instance: int):
        '''Obtain a zero-copy 'view' of a instance of the factorization.'''
        if instance >= self.nvec:
            raise IndexError(
                f'Index {instance} out of range (nvec: {self.nvec})'
            )
        fac = type(self)(self._tsrex_original, nvec=1)
        instance_factors = []
        for f in self.factors:
            if f.shape[-1] == 1:
                instance_factors.append(f[..., 0])
            else:
                instance_factors.append(f[..., instance])
        fac.factors = instance_factors
        return fac

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
        return self.tsrex | SlicingPropagator(full_idx) | ElementwiseEvaluator()

    def __getitem__(self, idx):
        '''Implements attribute-based access of factor tensors or output
        elements.'''
        if isinstance(idx, str):
            for n in dfs_filter(
                lambda n: n.name == 'tensor' and str(n.abstract.symbol) == idx,
                self.tsrex.root
            ):
                return n.data
            raise AttributeError(f'No factor tensor named {idx}.')
        else:
            return self._get_elements(idx)

    def __setitem__(self, name, data):
        '''Implements attribute-based access of factor tensors.'''
        for n in dfs_filter(
            lambda n: n.name == 'tensor' and str(n.abstract.symbol) == name,
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
                tensors=', '.join([str(n.abstract) for n in self.nodes]),
                pl='s' if len(self.nodes) > 1 else ''
            )

        def __getitem__(self, i):
            return getattr(self.nodes[i], self.attribute)

        def __setitem__(self, i, value):
            setattr(self.nodes[i], self.attribute, value)

        def __iter__(self):
            for n in self.nodes:
                yield getattr(n, self.attribute)
