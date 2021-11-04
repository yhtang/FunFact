#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
import tqdm
from funfact.optim._adam_np import Adam
from funfact.model._factorization import Factorization
from jax import grad, jit


def mse_loss(model, target):
    '''Mean squared error loss.'''
    return (np.square(np.subtract(model, target))).mean(axis=None)


def factorize(tsrex, target, lr=0.1, beta1=0.9, beta2=0.999,
              epsilon=1e-7, nsteps=10000):
    '''Gradient descent optimizer for functional factorizations.

    Parameters
    ----------
    tsrex: TsrEx
        A FunFact tensor expression.
    target
        Target data tensor
    lr
        Learning rate (default: 0.1)
    beta1
        First order moment (default: 0.9)
    beta2
        Second order moment (default: 0.999)
    epsilon
        default: 1e-7
    nsteps
        Number of steps in gradient descent (default:10000)
    '''

    @jit
    def loss(fac, target):
        return mse_loss(fac(), target)
    gradient = jit(grad(loss))

    fac = Factorization(tsrex)

    opt = Adam(fac, lr, beta1, beta2, epsilon)

    def progressbar(n):
        return tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=True
            )

    for step in progressbar(nsteps):
        opt.step(gradient(fac, target))

    return fac
