#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from funfact.backend import active_backend as ab


class Loss(ABC):
    '''Base class for loss functions.'''

    @abstractmethod
    def _loss(self, model, target):
        pass

    def __call__(self, model, target, reduction='mean', sum_vec=True,
                 **kwargs):
        if target.ndim == model.ndim - 1:
            if model.shape[:-1] != target.shape:
                raise ValueError(f'Target shape {target.shape} and '
                                 f'model shape {model.shape[:-1]} mismatch.')
            data_axis = [i for i in range(target.ndim)]
            target = target[..., None]
        elif target.ndim == model.ndim:
            data_axis = [i for i in range(target.ndim)]
            if model.shape != target.shape:
                raise ValueError(f'Target shape {target.shape} and '
                                 f'model shape {model.shape} mismatch.')
        else:
            raise ValueError(f'Target is {target.ndim} dimensional, while '
                             f'model is {model.ndim} dimensional.')
        if reduction == 'mean':
            _loss = (self._loss(model, target)).mean(axis=data_axis)
        elif reduction == 'sum':
            _loss = ab.sum(self._loss(model, target), axis=data_axis)
        else:
            raise SyntaxError(
                'The reduction operation should be either mean or sum, got '
                f'{reduction} instead.'
            )
        if sum_vec:
            return ab.sum(_loss, axis=None)
        else:
            return _loss


class MSE(Loss):

    def _loss(self, model, target):
        return ab.square(ab.subtract(model, target))


class L1(Loss):

    def _loss(self, model, target):
        return ab.abs(ab.subtract(model, target))


class KLDivergence(Loss):

    def _loss(self, model, target):
        return ab.multiply(target, ab.log(ab.divide(target, model)))


mse_loss = MSE()
l1_loss = L1()
kldiv_loss = KLDivergence()
