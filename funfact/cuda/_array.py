#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda.driver import (
    managed_empty,
    managed_zeros
)
from pycuda.driver import mem_attach_flags as ma_flags


class ManagedArray:

    @staticmethod
    def empty(size, dtype=np.float32, order='F'):
        return managed_empty(size, dtype, order, ma_flags.GLOBAL)

    @staticmethod
    def zeros(size, dtype=np.float32, order='F'):
        return managed_zeros(size, dtype, order, ma_flags.GLOBAL)

    @staticmethod
    def copy(arr, dtype=None, order=None):
        u = managed_empty(
            arr.shape,
            dtype or arr.dtype,
            order or ('C' if arr.flags.c_contiguous else 'F'),
            ma_flags.GLOBAL
        )
        u[:] = arr[:]
        return u
