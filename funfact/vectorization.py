#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang import index
from funfact.lang.interpreter import (
    IndexnessAnalyzer,
    TypeDeducer,
    Vectorizer
)


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
    return tsrex | IndexnessAnalyzer() \
                 | TypeDeducer() \
                 | Vectorizer(n, index().root, append)


def view(factors, fac_scalar, instance: int, append: bool = False):
    '''Obtain a zero-copy instance from a vectorized factorization model.

    Args:
        factors (list):
            Factor tensors in the vectorized factorization model.
        fac_scalar (Factorization):
            A factorization model created from the original, un-vectorized
            tensor expression
        instance (int):
            Index along the vectorization dimension.
        append (bool):
            Indicates whether the vectorization dimension was appended
            or prepended.

    Returns:
        Factorization:
            A factorization model.
    '''
    indices = tuple([..., instance] if append else [instance, ...])
    for i, f in enumerate(factors):
        fac_scalar.factors[i] = f[indices]
    return fac_scalar
