#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    LeafInitializer,
    PayloadMerger,
    IndexPropagator,
    Evaluator
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
    _evaluator = Evaluator()

    def __init__(self, tsrex, initialize=True):
        if initialize is True:
            self.tsrex = tsrex | self._leaf_initializer
        else:
            self.tsrex = tsrex

    @property
    def tsrex(self):
        return self._tsrex

    @tsrex.setter
    def tsrex(self, _tsrex):
        self._tsrex = _tsrex

    def __call__(self):
        '''Shorthand for :py:meth:`forward`.'''
        return self.forward()

    def forward(self):
        '''Evaluate the tensor expression the result.'''
        out, _ = (
            self.tsrex,
            self.tsrex | self._index_propagator
        ) | self._payload_merger | self._evaluator
        return out

    def getattr(self, tensor_name):
        '''Implements attribute-based access of factor tensors.'''
        raise NotImplementedError()

    @property
    def factors(self):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. For use with a gradient optimizer.'''
        raise NotImplementedError()
