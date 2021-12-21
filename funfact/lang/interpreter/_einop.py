#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import jax.numpy as np
# import numpy
import re
import numpy as np
from funfact.backend import active_backend as ab


def _einop(spec: str, lhs, rhs, reduction: str, pairwise: str):
    '''Einstein operation between two nd arrays.
    TODO: update documentation.
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
    out = re.fullmatch(r'(\w*),(\w*)->(\w*)\|(\w*)', spec)
    lhs_spec = list(out.group(1))
    rhs_spec = list(out.group(2))
    out_spec = list(out.group(3))
    kron_spec = list(out.group(4))

    # reorder lhs and rhs in alphabetical order
    if lhs_spec:
        lhs = ab.transpose(lhs, np.argsort(lhs_spec))
    if rhs_spec:
        rhs = ab.transpose(rhs, np.argsort(rhs_spec))

    # determine all indices in alphabetical order
    indices_all = set(lhs_spec).union(rhs_spec)
    indices_all = sorted(list(indices_all))

    # determine dimensions of lhs and rhs tensors,
    # contraction axis, Kronecker indices and remaining indices
    shape_lhs = lhs.shape
    shape_rhs = rhs.shape
    j_l = 0
    j_r = 0
    i = 0
    dim_lhs = []
    dim_rhs = []
    con_ax = []
    indices_rem = []
    kron_res = []
    for c in indices_all:
        if c in out_spec:
            # non-contracting index
            indices_rem.append(c)
            if c in kron_spec:
                dim_lhs.append(None)
                dim_lhs.append(slice(None))
                dim_rhs.append(slice(None))
                dim_rhs.append(None)
                kron_res.append(shape_lhs[j_l]*shape_rhs[j_r])
                i += 2
                j_l += 1
                j_r += 1
            else:
                if c in lhs_spec and c in rhs_spec:
                    dim_lhs.append(slice(None))
                    dim_rhs.append(slice(None))
                    if shape_lhs[j_l] > shape_rhs[j_r]:
                        kron_res.append(shape_lhs[j_l])
                    else:
                        kron_res.append(shape_rhs[j_r])
                    j_l += 1
                    j_r += 1
                elif c in lhs_spec:
                    dim_lhs.append(slice(None))
                    dim_rhs.append(None)
                    kron_res.append(shape_lhs[j_l])
                    j_l += 1
                elif c in rhs_spec:
                    dim_lhs.append(None)
                    dim_rhs.append(slice(None))
                    kron_res.append(shape_rhs[j_r])
                    j_r += 1
                i += 1
        else:
            # contracting index
            dim_lhs.append(slice(None))
            dim_rhs.append(slice(None))
            con_ax.append(i)
            j_l += 1
            j_r += 1
            i += 1

    # compute the contraction in alphabetical order
    op_redu = getattr(ab, reduction)
    op_pair = getattr(ab, pairwise)

    # reorder contraction according to out_spec
    dictionary = dict(zip(indices_rem, np.arange(len(indices_rem))))
    res_order = [dictionary[key] for key in out_spec]

    return ab.transpose(
        ab.reshape(
            op_redu(
                op_pair(
                    lhs[tuple(dim_lhs)],
                    rhs[tuple(dim_rhs)]
                ),
                axis=tuple(con_ax)
            ),
            tuple(kron_res),
            order='F'
        ),
        res_order
    )
