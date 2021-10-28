#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    depth_first_apply,
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

    def getattr(self, tensor_name):
        '''Implements attribute-based access of factor tensors.'''
        raise NotImplementedError()

    @property
    def factors(self):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. For use with a gradient optimizer.'''
        def get_data(n):
            if n.data is not None:
                yield n.data

        return list(depth_first_apply(self.tsrex.root, get_data, True))
