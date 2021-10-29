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

    def __getattr__(self, tensor_name):
        '''Implements attribute-based access of factor tensors.'''
        for n in dfs_filter(
            lambda n: n.name == 'tensor' and n.value.symbol == tensor_name,
            self.tsrex.root
        ):
            return n.data
        raise AttributeError(f'No factor tensor named {tensor_name}.')

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
