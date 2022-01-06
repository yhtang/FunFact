#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang import index
from funfact.lang.interpreter import (
    IndexAnalyzer,
    LeafVectorizer,
    EinopVectorizer,
)
from funfact.model import Factorization


def vectorize(tsrex, n, post: bool = True):
    ''''Vectorize' a tensor expression by extending its dimensionality by one.
    Each slice along the vectorization dimension of a factorization model
    represents an independent realization of the original tensor expression.
    A typical use case is multi-replica optimization.

    The dimensionality of each leaf (tensor) node of the expression will be
    increments using the following rule:

    - If the tensor has a random initializer, then its shape is simply
    extended, i.e. `shape_vectorized = (*shape_vectorized, n)`.
    - If the tensor has a concrete initializer, then a broadcasting dimension
    will be appended to the initializer, i.e.
    `initializer_vectorized = initializer_vectorized[..., None]`.

    Args:
        n (int > 0):
            Size of the vectorization dimension.
        post (post or pre order boolean) TODO

    Returns
        TsrEx:
            A vectorized tensor expression.
    '''
    i = index().root
    return tsrex | LeafVectorizer(n, i, post) | IndexAnalyzer() \
                 | EinopVectorizer(i, post)


def view(fac, tsrex_scalar, instance: int):
    '''Obtain a zero-copy instance from a vectorized factorization model.

    Args:
        fac (Factorization):
            A factorization model created from a vectorized tensor expresion.
        tsrex_scalar (TsrEx):
            The original, un-vectorized tensor expression.
        instance (int):
            Index along the vectorization dimension.

    Returns:
        Factorization:
            A factorization model.
    '''
    if instance >= fac.shape[-1]:
        raise IndexError(
            f'Only {fac.shape[-1]} vector instances exist, '
            f'index {instance} out of range.'
        )
    fac_scalar = Factorization(tsrex_scalar)
    for i, f in enumerate(fac.all_factors):
        fac_scalar.all_factors[i] = f[..., instance]
    return fac_scalar
