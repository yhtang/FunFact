#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
import numpy
import re

''' Einstein operation between two nd arrays. The specification string is assumed in
    the following format:
        abc,ad->bcd
    The order of the indices can be changed but the output indices should match with the
    contracted input indices.
    It is similar to explicit mode in einsum with that difference that we don't (yet) allow
    for surviving indices.
'''
def _einop(spec: str, lhs, rhs, op):    
    # parse input spec string
    lhs_spec, rhs_spec, res_spec = re.split(r'\W+', spec)    
    lhs_spec = list(lhs_spec)
    rhs_spec = list(rhs_spec)
    res_spec = list(res_spec)
    
    # reorder lhs and rhs in alphabetical order
    lhs_order = numpy.argsort(lhs_spec)
    lhs = np.transpose(lhs, lhs_order)
    rhs_order = numpy.argsort(rhs_spec)
    rhs = np.transpose(rhs, rhs_order)
    
    # determine all indices in alphabetical order
    indices_all = set(lhs_spec).union(rhs_spec)
    indices_all = sorted(list(indices_all))
    
    # determine dimensions of lhs and rhs tensors,
    # contraction axis, and remaining indices
    dim_lhs = []
    dim_rhs = []
    con_ax = []
    indices_rem = []
    for i, c in enumerate(indices_all):
        if c in lhs_spec:
            dim_lhs.append(slice(None))
        else:
            dim_lhs.append(None)
        if c in rhs_spec:
            dim_rhs.append(slice(None))
        else:
            dim_rhs.append(None)
        if c in lhs_spec and c in rhs_spec:
            con_ax.append(i)
        if (c in lhs_spec) ^ (c in rhs_spec):
            indices_rem.append(c)
            
    dim_lhs = tuple(dim_lhs)
    dim_rhs = tuple(dim_rhs)
    con_ax = tuple(con_ax)
    
    # compute the contraction in alphabetical order
    result = np.sum(op(lhs[dim_lhs], rhs[dim_rhs]), axis=con_ax)
    
    # reorder contraction according to res_spec
    dictionary = dict(zip(indices_rem, numpy.arange(len(indices_rem))))
    res_order = [dictionary[key] for key in res_spec]
    return np.transpose(result, res_order)