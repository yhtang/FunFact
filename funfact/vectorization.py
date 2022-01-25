#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang import index
from funfact.lang.interpreter import IndexnessAnalyzer, Compiler, Vectorizer
from funfact.model import Factorization


def vectorize(tsrex, n, append: bool = False):
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
        append (bool):
            If True, the vectorizing index is set to the last index of every
            leaf. If False, the vectorizing index is set to the first index
            of every leaf.

    Returns
        TsrEx:
            A vectorized tensor expression.
    '''
    i = index().root
    return tsrex | IndexnessAnalyzer() | Compiler() | Vectorizer(n, i, append)


def view(fac, tsrex_scalar, instance: int, append: bool = False):
    '''Obtain a zero-copy instance from a vectorized factorization model.

    Args:
        fac (Factorization):
            A factorization model created from a vectorized tensor expresion.
        tsrex_scalar (TsrEx):
            The original, un-vectorized tensor expression.
        instance (int):
            Index along the vectorization dimension.
        append (bool):
            Indicates whether the vectorization dimension was appended
            or prepended.

    Returns:
        Factorization:
            A factorization model.
    '''
    nvec = fac.shape[-1] if append else fac.shape[0]
    if instance >= nvec:
        raise IndexError(
            f'Only {nvec} vector instances exist, '
            f'index {instance} out of range.'
        )
    fac_scalar = Factorization(tsrex_scalar)
    instance = [..., instance] if append else [instance, ...]
    for i, f in enumerate(fac.all_factors):
        fac_scalar.all_factors[i] = f[tuple(instance)]
    return fac_scalar
