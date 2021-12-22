#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter import (
    IndexPropagator,
    Vectorizer
)
from funfact.model import Factorization


def vectorize(tsrex, n):
    '''

    Args:
        nvec (int > 0):
            Number of parallel random instances to create.
    '''
    return tsrex | IndexPropagator() | Vectorizer(n)


def view(fac, tsrex_scalar, instance: int):
    '''Obtain a zero-copy 'view' of a instance of the factorization.'''
    fac_scalar = Factorization(tsrex_scalar)
    for i, f in enumerate(fac.factors):
        try:
            fac_scalar.factors[i] = f[..., instance]
        except IndexError:
            # if the last dimension is a broadcasting one
            fac_scalar.factors[i] = f[..., 0]
    return fac_scalar
