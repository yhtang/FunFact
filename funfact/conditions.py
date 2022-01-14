#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from funfact import active_backend as ab


class _Condition(ABC):
    '''Base class for all conditions on leaf tensors.'''

    def __init__(self, weight=1.0, elementwise='mse', reduction='mean'):
        '''Condition on leaf tensor.

        Args:
            weight (float): Hyperparameter that determines weight of condition.
            elementwise:
                If 'mse', the mean-squared error is evaluated elementwise.
                If 'l1', the L1 error is evaluated elementwise.
            reduction:
                If 'mean', the mean of the elementwise condition is returned.
                If 'sum', the sum over the elementwise condition is returned.
        '''
        self.weight = weight
        if elementwise == 'mse':
            def elementwise(data):
                return ab.square(ab.abs(data))
        elif elementwise == 'l1':
            def elementwise(data):
                return ab.abs(data)
        else:
            raise SyntaxError(
                'The elementwise function should be either mse or l1, '
                f'get {elementwise} instead.'
            )
        self.elementwise = elementwise
        if reduction == 'mean':
            def reduction(data):
                return ab.mean(data)
        elif reduction == 'sum':
            def reduction(data):
                return ab.sum(data)
        else:
            raise SyntaxError(
                'The reduction operation should be either mean or sum, '
                f'got {self.reduction} instead.'
            )
        self.reduction = reduction

    @abstractmethod
    def _condition(self, data):
        pass

    def __call__(self, data, sum_vec=None):
        '''Evaluate condition on leaf tensor.

        Args:
            data (tensor): leaf tensor
        '''
        return self.weight * self.reduction(
            self.elementwise(self._condition(data))
        )


class _MatrixCondition(_Condition):
    '''Base class for all conditions only to be evaluated for matrices.'''

    def __call__(self, data, sum_vec=None):
        if data.ndim != 2:
            raise ValueError('Penalty can only be evaluated for matrices, '
                             f'got tensor with shape {data.shape}.')
        return super().__call__(data)


class UpperTriangular(_MatrixCondition):
    '''Checks an upper triangular condition.'''

    def _condition(self, data):
        return ab.tril(data, -1)


class Unitary(_MatrixCondition):
    '''Checks a unitary condition.'''

    def _condition(self, data):
        return ab.subtract(
            ab.matmul(data, ab.conj(ab.transpose(data, (1, 0)))),
            ab.eye(data.shape[0])
        )


class Diagonal(_MatrixCondition):
    '''Checks a diagonal condition.'''

    def _condition(self, data):
        return ab.add(ab.triu(data, 1), ab.tril(data, -1))


class NonNegative(_Condition):
    '''Checks a non-negative condition.'''

    def _condition(self, data):
        negative = data[data < 0.0]
        return negative if ab.any(negative) else \
            ab.tensor(0.0)


class NoCondition(_Condition):
    '''No condition enforced.'''

    def _condition(self, data):
        return ab.tensor(0.0)


def vmap(condition, append: bool = True):
    '''Vectorizes a condtion.

    Args:
        condition (callable): non-vectorized condition.
        append (bool):
            If True, the last index of shape is considered the vectorizing
            index. If False, the first index of shape tuple is considered
            the vectorizing index.
    '''
    def wrapper(data, sum_vec: bool = True):
        '''Vectorized condition

        Args:
            data (tensor): vectorized tensor leaf.
            sum_vec (bool):
                If True, the conditions are summed over the vectorizing
                dimension. If False, the conditions per instance in the
                vectorized model are returned.
        '''
        shape = data.shape
        nvec = shape[-1] if append else shape[0]

        def _get_instance(i):
            return data[..., i] if append else data[i, ...]

        conditions = ab.stack([condition(_get_instance(i)) for i in
                               range(nvec)], 0)
        return ab.sum(conditions) if sum_vec else conditions
    return wrapper
