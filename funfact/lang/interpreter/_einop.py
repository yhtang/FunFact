#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as np
import numpy
import re


def log_add_exp(lhs, rhs):
    return np.log(np.add(np.exp(lhs), np.exp(rhs)))


def log_sum_exp(data, axis=None):
    return np.log(np.sum(np.exp(data), axis=axis))


class DummyBackend:

    add = np.add
    sub = np.subtract
    mul = np.multiply
    div = np.divide
    log_add_exp = log_add_exp
    min = np.min
    max = np.max
    sum = np.sum
    log_sum_exp = log_sum_exp


def _einop(spec: str, lhs, rhs, reduction: str, pairwise: str):
    '''Einstein operation between two nd arrays.

    Parameters
    ----------
    spec: str
        Specification string for index contraction. The following
        options are supported:
            'abc,ad->bcd', 'abc,ad->cbd', ....
            'abc,ad->abcd', 'abc,ad->bacd', ...
            'abc,ad->cd', 'abc,ad->bd', ...
            'abc,ad->b', 'abc,ad->c', ...
            'abc,ad->'
    lhs: array
        Left hand side array
    rhs: array
        Right hand side array
    reduction: str
        Name of the reduction operator
    pairwise: str
        Name of the pairwise operator
    '''

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
        if c not in res_spec:
            con_ax.append(i)
        else:
            indices_rem.append(c)

    dim_lhs = tuple(dim_lhs)
    dim_rhs = tuple(dim_rhs)
    con_ax = tuple(con_ax)

    # compute the contraction in alphabetical order
    op_redu = getattr(DummyBackend, reduction)
    op_pair = getattr(DummyBackend, pairwise)

    result = op_redu(op_pair(lhs[dim_lhs], rhs[dim_rhs]), axis=con_ax)

    # reorder contraction according to res_spec
    dictionary = dict(zip(indices_rem, numpy.arange(len(indices_rem))))
    res_order = [dictionary[key] for key in res_spec]
    return np.transpose(result, res_order)
