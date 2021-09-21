#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pycuda.compiler import SourceModule
import funfact.cpp


def jit(source, name, *compiler_options, **pycuda_options):
    return SourceModule(
        source,
        options=['-std=c++14',
                 '-O4',
                 '--use_fast_math',
                 '--expt-relaxed-constexpr',
                 '--maxrregcount=64',
                 # '-Xptxas', '-v',
                 '-lineinfo',
                 *compiler_options],
        include_dirs=funfact.cpp.__path__,
        no_extern_c=True,
        # keep=True,
        **pycuda_options
    ).get_function(name)
