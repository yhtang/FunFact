#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda.compiler import SourceModule

import symfac.cpp
from symfac.cuda import ManagedArray


class Adam:

    @classmethod
    def get_kernel(cls):
        try:
            return cls._kernel
        except AttributeError:
            cls._kernel = SourceModule(
                open(symfac.cpp.__path__[0] + '/adam.cu').read(),
                options=['-std=c++14',
                         '-O4',
                         '--use_fast_math',
                         '--expt-relaxed-constexpr',
                         '--maxrregcount=64',
                         '-Xptxas', '-v',
                         '-lineinfo'],
                no_extern_c=True
            ).get_function('adam')

            return cls._kernel

    @property
    def kernel(self):
        return type(self).get_kernel()

    def __init__(
        self, x, lr, beta1=0.9, beta2=0.999, epsilon=1e-7
    ):
        self.x = x
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.M = [ManagedArray.zeros(w.shape, np.float32) for w in x]
        self.V = [ManagedArray.zeros(w.shape, np.float32) for w in x]

    def step(self, grad):
        for w, dw, m, v in zip(self.x, grad, self.M, self.V):
            n = np.prod(w.shape).item()
            self.kernel(
                w, dw, m, v,
                np.float32(self.lr),
                np.float32(self.beta1),
                np.float32(self.beta2),
                np.float32(self.epsilon),
                np.int32(n),
                block=(1024, 1, 1),
                grid=((n + 1023) // 1024, 1, 1)
            )
