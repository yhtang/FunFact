#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    dfs_filter,
    EinsteinSpecGenerator,
    Evaluator,
    LeafInitializer,
    PayloadMerger,
    IndexPropagator,
    ElementwiseEvaluator,
    SlicingPropagator,
    ShapeAnalyzer,
    Vectorizer,
    Devectorizer
)
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Factorization:
    '''A factorization model is a concrete realization of a tensor expression.
    The factor tensors of the model can be optimized to obtain a better
    approximation of a target tensor.

    Parameters
    ----------
    tsrex: TsrEx
        A FunFact tensor expression.
    '''

    _leaf_initializer = LeafInitializer()
    _payload_merger = PayloadMerger()
    _index_propagator = IndexPropagator()
    _einspec_generator = EinsteinSpecGenerator()
    _shape_analyzer = ShapeAnalyzer()
    _evaluator = Evaluator()
    _elementwise_evaluator = ElementwiseEvaluator()

    def __init__(self, tsrex, initialize=True, nvec=1):
        tsrex = tsrex | self._index_propagator
        if nvec > 1:
            tsrex = tsrex | Vectorizer(nvec)
            tsrex = tsrex | self._index_propagator
        if initialize is True:
            tsrex = tsrex | self._leaf_initializer
        tsrex = tsrex | self._shape_analyzer | self._einspec_generator
        self._tsrex = tsrex
        self._nvec = nvec

    @property
    def tsrex(self):
        return self._tsrex

    @property
    def nvec(self):
        return self._nvec

    @property
    def shape(self):
        return self._tsrex.shape

    @property
    def ndim(self):
        return self._tsrex.ndim

    def devectorize(self, instance: int):
        '''Devectorize a factorizaition and keep a single instance.'''
        if instance >= self.nvec:
            raise IndexError(
                f'Index {instance} out of range (nvec: {self.nvec})'
            )
        tsrex = self.tsrex | Devectorizer(instance)
        return type(self)(tsrex, initialize=False)

    def __call__(self):
        '''Shorthand for :py:meth:`forward`.'''
        return self.forward()

    def forward(self):
        '''Evaluate the tensor expression the result.'''
        return self.tsrex | self._evaluator

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
            if idx.start >= self.shape[i] or idx.start < -self.shape[i]:
                raise IndexError(
                    f'index.start {idx.start} is out of bounds for axis {i} '
                    f'with size {self.shape[i]}'
                )
            if idx.stop > self.shape[i] or idx.stop <= -self.shape[i]:
                raise IndexError(
                    f'index.stop {idx.stop} is out of bounds for axis {i} '
                    f'with size {self.shape[i]}'
                )

        # Evaluate model
        _index_slicer = SlicingPropagator(full_idx)
        return self.tsrex | _index_slicer | self._elementwise_evaluator

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

    @property
    def factors(self):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. For use with a gradient optimizer.'''
        return [
            n.data for n in dfs_filter(
                lambda n: n.name == 'tensor', self.tsrex.root
            )
        ]

    @factors.setter
    def factors(self, tensors):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. For use with a gradient optimizer.'''
        for i, n in enumerate(
            dfs_filter(lambda n: n.name == 'tensor', self.tsrex.root)
        ):
            n.data = tensors[i]

    def tree_flatten(self):
        return self.factors, self.tsrex

    @classmethod
    def tree_unflatten(cls, tsrex, tensors):
        unflatten = cls(tsrex, initialize=False)
        unflatten.factors = tensors
        return unflatten
