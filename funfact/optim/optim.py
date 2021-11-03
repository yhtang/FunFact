#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
import tqdm
from funfact.optim._adam_np import Adam
from funfact.model._factorization import Factorization
from jax import grad, jit

def mse_loss(model, target):
    return (np.square(np.subtract(model, target))).mean(axis=None)


def factorize(tsrex, target, lr=0.1, beta1=0.9, beta2=0.999, 
              epsilon=1e-7, nsteps=10000
    ):
    '''
    Gradient descent optimizer for functional factorizations.
    Input: 
        tsrex to optimize
        optimization target data
    '''
    
    @jit
    def loss(fac, target):
        return mse_loss(fac(), target)
    gradient = jit(grad(loss))

    fac = Factorization(tsrex)

    opt = Adam(fac, lr, beta1, beta2, epsilon)

    progressbar = lambda n: tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=True
            )
    
    for step in progressbar(nsteps):
        opt.step(gradient(fac,target))

    return fac
