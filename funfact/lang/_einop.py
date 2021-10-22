#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

'''
    Einstein operation between two nd arrays. The specification
    string 'spec' is expected to be ordered alphabetically for the
    left hand side.
'''
def _einop(spec: str, lhs, rhs, op):
    # parse specification string
    lhs_spec, rhs_spec = spec.split(',')
    # transpose rhs
    rhs = np.transpose(rhs, sorted(range(len(rhs_spec)), key=rhs_spec.__getitem__))
    # determine indices
    indices_all = set(lhs_spec).union(rhs_spec)
    indices_all = sorted(list(indices_all))
    indices_lhs = []
    indices_rhs = []
    con_axis = []
    for i, c in enumerate(indices_all):
        if c in lhs_spec:
            indices_lhs.append(slice(None))
        else:
            indices_lhs.append(None)
        if c in rhs_spec:
            indices_rhs.append(slice(None))
        else:
            indices_rhs.append(None)
        if c in lhs_spec and c in rhs_spec:
            con_axis.append(i)
    indices_lhs = tuple(indices_lhs)
    indices_rhs = tuple(indices_rhs)
    con_axis = tuple(con_axis)
    # compute contraction
    return np.sum(op(lhs[indices_lhs], rhs[indices_rhs]), axis=con_axis)
    