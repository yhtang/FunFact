#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    dfs_filter,
    EinsteinSpecGenerator,
    Evaluator,
    LeafInitializer,
    PayloadMerger,
    IndexPropagator
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
    _evaluator = Evaluator()

    def __init__(self, tsrex, initialize=True):
        self._tsrex = (
            (tsrex | self._leaf_initializer) if initialize is True else tsrex,
            (tsrex | self._index_propagator | self._einspec_generator)
        ) | self._payload_merger

    @property
    def tsrex(self):
        return self._tsrex

    def __call__(self):
        '''Shorthand for :py:meth:`forward`.'''
        return self.forward()

    def forward(self):
        '''Evaluate the tensor expression the result.'''
        return self.tsrex | self._evaluator

    def __getitem__(self, idx):
        '''Implements attribute-based access of factor tensors or output
        elements.'''
        if isinstance(idx, str):
            for n in dfs_filter(
                lambda n: n.name == 'tensor' and n.value.symbol == idx,
                self.tsrex.root
            ):
                return n.data
            raise AttributeError(f'No factor tensor named {idx}.')
        elif isinstance(idx, int):
            raise NotImplementedError()
        elif isinstance(idx, slice):
            raise NotImplementedError()
        elif isinstance(idx, tuple):
            raise NotImplementedError()

    def __setitem__(self, name, data):
        '''Implements attribute-based access of factor tensors.'''
        for n in dfs_filter(
            lambda n: n.name == 'tensor' and n.value.symbol == name,
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