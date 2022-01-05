#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Number
from funfact import active_backend as ab


class Zeros:
    '''Set all elements to 0.

    Args:
        dtype: Numerical type of elements.
    '''

    def __init__(self, dtype=None):
        self.dtype = dtype or ab.float32

    def __call__(self, shape):
        return ab.zeros(shape, self.dtype)


class Ones:
    '''Set all elements to 1.

    Args:
        dtype: Numerical type of elements.
    '''

    def __init__(self, dtype=None):
        self.dtype = dtype or ab.float32

    def __call__(self, shape):
        ab.ones(shape, self.dtype)


class Normal:
    '''Sample elements from i.i.d. normal distributions.

    Args:
        std: Standard deviation of the distribution.
        truncation:

            - If `True`, clamp values at twice the standard deviation.
            - If `False`, no truncation happens.
            - If number, clamp values at the specified multiple of standard
            deviation

        dtype: numerical type of elements.
    '''

    def __init__(self, std=0.01, truncation=False, dtype=None):
        self.std = std
        self.dtype = dtype or ab.float32
        if truncation is True:
            self.truncation = 2.0 * std
        elif isinstance(truncation, Number):
            self.truncation = float(truncation) * std
        else:
            self.truncation = 0

    def __call__(self, shape):
        n = ab.normal(0.0, self.std, shape, dtype=self.dtype)
        if self.truncation:
            n = ab.maximum(-self.truncation, ab.minimun(self.truncation, n))
        return n


class Uniform:
    '''Sample elements from the uniform distributions.

    Args:
        scale: Upper bound of the distribution. Lower bound is always 0.
        dtype: numerical type of elements.
    '''

    def __init__(self, scale=0.01, dtype=None):
        self.scale = scale
        self.dtype = dtype or ab.float32

    def __call__(self, shape):
        return self.scale * ab.uniform(shape, dtype=self.dtype)


class VarianceScaling:
    '''Initializes with adaptive scale according to the shape.

    Args:
        scale: Scaling factor (positive float).
        distribution: 'truncated' or 'normal' or 'uniform'.

            - If `'normal'`, draw from a zero-mean normal distribution with
            standard deviation `sqrt(scale / n)`, where `n` is the
            dimensionality of `axis`.

            - If `'truncated'`, the absolute values of the samples are
            truncated below 2 standard deviations before truncation.

            - If `'uniform'`, samples are drawn from:
                - a uniform interval, if `dtype` is real
                - a uniform disk, if `dtype` is complex with a mean of zero
                and a standard deviation of `scale`.

        axis: dimension of the given shape.
        dtype: numerical type of elements.
    '''
    def __init__(
        self, scale=0.01, distribution='normal', axis=-1, dtype=None
    ):
        self.scale = scale
        self.axis = axis
        self.dtype = dtype or ab.float32
        if distribution == 'normal':
            self.distribution = Normal(
                1.0, truncation=False, dtype=dtype
            )
        elif distribution == 'truncated':
            self.distribution = Normal(
                1.0, truncation=2, dtype=dtype
            )
        elif distribution == 'uniform':
            self.distribution = Uniform(
                1.0, dtype=dtype
            )
        else:
            raise ValueError(f'Invalid distribution: {distribution}.')

    def __call__(self, shape):
        std = (self.scale / shape[self.axis])**0.5
        return std * self.distribution.init(shape)
