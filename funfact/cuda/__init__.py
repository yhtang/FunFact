#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pycuda.driver as cuda
from ._array import ManagedArray
from ._context import context_manager
from ._jit import jit


_cuda_initialized = False
if not _cuda_initialized:
    cuda.init()
    _cuda_initialized = True


__all__ = ['jit', 'context_manager', 'ManagedArray']
