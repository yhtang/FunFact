#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np


class Adam:

    def __init__(
        self, x, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-7
    ):
        self.x = x
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.M = [np.zeros(w.shape) for w in x.factors]
        self.V = [np.zeros(w.shape) for w in x.factors]

    def step(self, grad):
        grad_factors = grad.factors
        new_factors = self.x.factors
        for i in range(len(self.x.factors)):
            self.M[i] = self.beta1 * self.M[i] + \
                (1 - self.beta1) * grad_factors[i]
            self.V[i] = self.beta2 * self.V[i] + \
                (1 - self.beta2) * grad_factors[i] * grad_factors[i]
            mhat = self.M[i] / (1 - self.beta1)
            vhat = self.V[i] / (1 - self.beta2)
            new_factors[i] -= self.lr * np.reciprocal(
                            np.sqrt(vhat + self.epsilon)) * mhat
        self.x.factors = new_factors
