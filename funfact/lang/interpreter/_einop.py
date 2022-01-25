#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import jax.numpy as np
# import numpy
import re
import numpy as np
from funfact.backend import active_backend as ab
from funfact.util.set import ordered_symmdiff


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
    out = re.fullmatch(r'(\w*),(\w*)(->(\w*))?(\|(\w*))?', spec)
    lhs_spec = list(out.group(1))
    rhs_spec = list(out.group(2))
    out_spec = list(out.group(4) or ordered_symmdiff(lhs_spec, rhs_spec))
    kron_spec = list(out.group(6) or [])

    # reorder lhs and rhs in alphabetical order
    if lhs_spec:
        lhs = ab.transpose(lhs, np.argsort(lhs_spec))
    if rhs_spec:
        rhs = ab.transpose(rhs, np.argsort(rhs_spec))

    # determine all indices in alphabetical order
    indices_all = sorted(set(lhs_spec).union(rhs_spec))

    # determine dimensions of lhs and rhs tensors,
    # contraction axis, Kronecker indices and remaining indices
    j_l = 0
    j_r = 0
    dim_lhs = []
    dim_rhs = []
    con_ax = []
    indices_rem = []
    kron_res = []

    colon = slice(None)
    newaxis = None

    for c in indices_all:
        if c in out_spec:
            # non-contracting index
            indices_rem.append(c)
            if c in kron_spec:
                dim_lhs += [newaxis, colon]
                dim_rhs += [colon, newaxis]
                kron_res.append(lhs.shape[j_l] * rhs.shape[j_r])
                j_l += 1
                j_r += 1
            else:
                if c in lhs_spec and c in rhs_spec:
                    dim_lhs.append(colon)
                    dim_rhs.append(colon)
                    kron_res.append(max(lhs.shape[j_l], rhs.shape[j_r]))
                    j_l += 1
                    j_r += 1
                elif c in lhs_spec:
                    dim_lhs.append(colon)
                    dim_rhs.append(newaxis)
                    kron_res.append(lhs.shape[j_l])
                    j_l += 1
                elif c in rhs_spec:
                    dim_lhs.append(newaxis)
                    dim_rhs.append(colon)
                    kron_res.append(rhs.shape[j_r])
                    j_r += 1
        else:
            # contracting index
            con_ax.append(len(dim_lhs))
            dim_lhs.append(colon)
            dim_rhs.append(colon)
            j_l += 1
            j_r += 1

    # compute the contraction in alphabetical order
    op_redu = getattr(ab, reduction)
    op_pair = getattr(ab, pairwise)

    def op_redu_overload(tensor):
        return op_redu(tensor, tuple(con_ax)) if con_ax else tensor

    # reorder contraction according to out_spec
    res_order = [indices_rem.index(i) for i in out_spec]

    return ab.transpose(
        ab.reshape(
            op_redu_overload(
                op_pair(
                    lhs[tuple(dim_lhs)],
                    rhs[tuple(dim_rhs)]
                ),
            ),
            tuple(kron_res),
            order='F'
        ),
        res_order
    )
