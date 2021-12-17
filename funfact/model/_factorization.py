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
    Vectorizer
)
from funfact.backend import active_backend as ab


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
        self._otsrex = tsrex
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

    def view(self, instance: int):
        '''Create a view on a certain instance of the factorization.'''
        if instance >= self.nvec:
            raise IndexError(
                f'Index {instance} out of range (nvec: {self.nvec})'
            )
        fac = type(self)(self._otsrex, initialize=False)
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

    @property
    def factors(self):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. For use with a gradient optimizer.'''
        return self._NodeView(
            'data',
            list(dfs_filter(lambda n: n.name == 'tensor', self.tsrex.root))
        )

    @factors.setter
    def factors(self, tensors):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. For use with a gradient optimizer.'''
        for i, n in enumerate(
            dfs_filter(lambda n: n.name == 'tensor', self.tsrex.root)
        ):
            n.data = tensors[i]


@ab.autograd_decorator
class AutoGradFactorization(Factorization, ab.AutoGradMixin):
    pass
