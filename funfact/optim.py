#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab


class Adam:

    def __init__(
        self, X, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-7
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
