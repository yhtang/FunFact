#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np


class Loss:

    def __init__(self, type='mse'):
        if type == 'mse':
            def _loss(model, target):
                return np.square(np.subtract(model, target))
        elif type == 'l1':
            def _loss(model, target):
                return np.abs(np.subtract(model, target))
        elif type == 'kl_divergence':
            def _loss(model, target):
                return np.multiply(target, np.log(np.divide(target, model)))
        else:
            raise ValueError(f'Unsupported loss function type {type}')
        self._loss = _loss

    def __call__(self, model, target, reduction='mean', sum_vec=True):
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
            _loss = np.sum(self._loss(model, target), axis=data_axis)
        else:
            raise SyntaxError(
                'The reduction operation should be either mean or sum, got '
                f'{reduction} instead.'
            )
        if sum_vec:
            return np.sum(_loss, axis=None)
        else:
            return _loss
