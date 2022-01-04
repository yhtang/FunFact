#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from numbers import Number
from funfact import active_backend as ab


class _Initializer(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    @abstractmethod
    def init(self, *args, **kwargs):
        pass


class Zeros(_Initializer):

    def __init__(self, dtype=None):
        self.dtype = dtype or ab.float32

    def init(self, shape):
        return ab.zeros(shape, self.dtype)


class Ones(_Initializer):

    def __init__(self, dtype=None):
        self.dtype = dtype or ab.float32

    def init(self, shape):
        ab.ones(shape, self.dtype)


class Normal(_Initializer):

    def __init__(self, std=0.01, truncation=False, dtype=None):
        self.std = std
        self.dtype = dtype or ab.float32
        if truncation is True:
            self.truncation = 2.0 * std
        elif isinstance(truncation, Number):
            self.truncation = float(truncation) * std
        else:
            self.truncation = 0

    def init(self, shape):
        n = ab.normal(0.0, self.std, shape, dtype=self.dtype)
        if self.truncation:
            n = ab.maximum(-self.truncation, ab.minimun(self.truncation, n))
        return n


class Uniform(_Initializer):

    def __init__(self, scale=0.01, dtype=None):
        self.scale = scale
        self.dtype = dtype or ab.float32

    def init(self, shape):
        return self.scale * ab.uniform(shape, dtype=self.dtype)


class VarianceScaling(_Initializer):

    def __init__(
        self, mode, scale=0.01, distribution='normal', in_axis=-2, out_axis=-1,
        dtype=None
    ):
        assert mode in ['fan_in', 'fan_out', 'fan_avg']
        self.mode = mode
        self.scale = scale
        self.in_axis = in_axis
        self.out_axis = out_axis
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

    def init(self, shape):
        if self.mode == 'fan_out':
            std = (self.scale / shape[self.out_axis])**0.5
        elif self.mode == 'fan_in':
            std = (self.scale / shape[self.in_axis])**0.5
        elif self.mode == 'fan_avg':
            std = (
                self.scale / (0.5 * (shape[self.out_axis] + shape[self.in_axis]))
            )**0.5

        return std * self.distribution.init(shape)


zeros = Zeros()
ones = Ones()
normal = Normal()
uniform = Uniform()
variance_scaling = VarianceScaling('fan_avg')
