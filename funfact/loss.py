#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from funfact.backend import active_backend as ab


class Loss(ABC):
    '''Base class for FunFact loss functions.

    Steps to define a custom loss function:

    1. *inherit* from `Loss`
    2. implement `_loss` method to perform the elementwise evaluation for your
    loss function. The reduction is handled by this base class.

    Args:
        reduction (str):
            If 'mean', the mean of the elementwise loss is returned.
            If 'sum', the sum over the elementwise loss is returned.
        sum_vec (bool): If True, the loss is summed over the vectorizing
            dimension and a single loss value is returned. If False, the
            loss of every model instance is returned in an array.
        vectorized_along_last (bool): If True, the model is vectorized
            along the last dimension. If False, the model is assumed along
            the first dimension.
    '''

    @abstractmethod
    def _loss(self, model, target):
        '''Elementwise evaluation of loss.'''
        pass

    def __init__(
        self, reduction='mean'
    ):
        if reduction not in ['mean', 'sum']:
            raise SyntaxError(
                'The reduction operation should be either mean or sum, got '
                f'{reduction} instead.'
            )
        self.reduction = reduction

    def __call__(
        self, model, target, sum_vec=True, vectorized_along_last=False
    ):
        '''Evaluate the loss function.

        Args:
            model (tensor): model data.
            target (tensor): target data.

        Returns:
            tensor:
                The loss values as a scalar (non-vectorized data) or 1D vector
                (vectorized data).
        '''
        if target.ndim == model.ndim - 1:  # vectorized model
            model_shape = model.shape[:-1] if vectorized_along_last else \
                          model.shape[1:]
            if model_shape != target.shape:
                raise ValueError(f'Target shape {target.shape} and model '
                                 f'shape {model_shape} mismatch.')
            data_axis = tuple(i if vectorized_along_last else i+1
                              for i in range(target.ndim))
            target = target[..., None] if vectorized_along_last else \
                target[None, ...]
        elif target.ndim == model.ndim:  # non-vectorized model
            data_axis = tuple(i for i in range(target.ndim))
            if model.shape != target.shape:
                raise ValueError(f'Target shape {target.shape} and '
                                 f'model shape {model.shape} mismatch.')
        else:
            raise ValueError(f'Target is {target.ndim} dimensional, while '
                             f'model is {model.ndim} dimensional.')
        if self.reduction == 'mean':
            _loss = (self._loss(model, target)).mean(axis=data_axis)
        elif self.reduction == 'sum':
            _loss = ab.sum(self._loss(model, target), axis=data_axis)
        if sum_vec:
            return ab.sum(_loss)
        else:
            return _loss


class MatrixLoss(Loss):
    '''Base class for FunFact loss functions for matrices (2D tensors).'''
    def __call__(
        self, model, target, sum_vec=True, vectorized_along_last=False
    ):
        if target.ndim != 2:
            raise RuntimeError(f'Target is not a matrix, has {target.ndim} '
                               'dimensions.')
        if model.ndim == 3:  # vectorized model
            if vectorized_along_last:
                loss = ab.stack(
                    [self._loss(model[..., i], target) for i in
                     range(model.shape[-1])]
                )
            else:
                loss = ab.stack(
                    [self._loss(model[i, ...], target) for i in
                     range(model.shape[0])]
                )
            data_axis = (1, 2)
        if model.ndim == 2:  # non-vectorized model
            loss = self._loss(model, target)
            data_axis = (0, 1)
        if self.reduction == 'mean':
            loss = loss.mean(axis=data_axis)
        if self.reduction == 'sum':
            loss = ab.sum(loss, axis=data_axis)
        if sum_vec:
            return ab.sum(loss)
        else:
            return loss


class MSE(Loss):
    '''Mean-Squared Error (MSE) loss.'''
    def _loss(self, model, target):
        # ab.abs: to handle both real and complex numbers
        return ab.square(ab.abs(ab.subtract(model, target)))


class L1(Loss):
    '''L1 loss.'''
    def _loss(self, model, target):
        return ab.abs(ab.subtract(model, target))


class KLDivergence(Loss):
    '''KL Divergence loss.'''
    def _loss(self, model, target):
        return ab.multiply(target, ab.log(ab.divide(target, model)))


class PhaseInvariantMSE(MatrixLoss):
    '''MSE loss for matrices agnostic to global phase.'''
    def _loss(self, model, target):
        return ab.square(ab.abs(
            ab.transpose(model.conj(), axes=(1, 0)) @ target) -
            ab.eye(model.shape[0], target.shape[1])
        )


mse = MSE()
l1 = L1()
kl_divergence = KLDivergence()
