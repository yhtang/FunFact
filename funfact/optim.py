#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from funfact.backend import active_backend as ab


class Optimizer(ABC):
    '''A FunFact optimizer has to implement two methods: an initializer to set
    the model and optimizer parameters, and a step method.'''

    @abstractmethod
    def __init__(self, model, **kwargs):
        pass

    @abstractmethod
    def step(self, grad):
        pass


class Adam(Optimizer):
    '''Adam gradient descent optimizer.

    Parameters
    ----------
    lr
        Learning rate (default: 0.1)
    beta1
        First order moment (default: 0.9)
    beta2
        Second order moment (default: 0.999)
    epsilon
        default: 1e-7
    '''
    def __init__(
        self, X, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-7, **kwargs
    ):
        self.X = X
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.M = [ab.zeros_like(x) for x in X]
        self.V = [ab.zeros_like(x) for x in X]

    def step(self, grad):
        for i, g in enumerate(grad):
            self.M[i] = self.beta1 * self.M[i] + (1 - self.beta1) * g
            self.V[i] = self.beta2 * self.V[i] + (1 - self.beta2) * g * g
            mhat = self.M[i] / (1 - self.beta1)
            vhat = self.V[i] / (1 - self.beta2)
            self.X[i] -= self.lr * mhat * ab.reciprocal(ab.sqrt(
                vhat + self.epsilon
            ))


class RMSprop(Optimizer):
    '''RMSprop gradient descent optimizer with momentum.

    Parameters
    ----------

    '''
    def __init__(
        self, X, lr=0.1, alpha=0.99, epsilon=1e-8, weight_decay=0,
        momentum=0, centered=False, **kwargs
    ):
        self.X = X
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.V = [ab.zeros_like(x) for x in X]
        self.B = [ab.zeros_like(x) for x in X]
        self.G = [ab.zeros_like(x) for x in X]

    def step(self, grad):
        if self.weight_decay != 0:
            for g, x in zip(grad, self.X):
                g += self.weight_decay * x
        for i, g in enumerate(grad):
            self.V[i] = self.alpha * self.V[i] + \
                        (1 - self.alpha) * g * g
            vhat = self.V[i]
            if self.centered:
                self.G[i] = self.alpha * self.G[i] + \
                            (1 - self.alpha) * g
                vhat -= self.G[i] * self.G[i]
            if self.momentum > 0:
                self.B[i] = self.momentum * self.B[i] + \
                            g * ab.reciprocal(
                            ab.sqrt(vhat) + self.epsilon)
                self.X[i] -= self.lr * self.B[i]
            else:
                self.X[i] -= self.lr * g * ab.reciprocal(
                                  ab.sqrt(vhat) + self.epsilon)
