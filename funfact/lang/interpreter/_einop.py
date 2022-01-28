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

    # move all surviving indices to the front and sort the rest
    indices_all = out_spec + sorted(
        set(lhs_spec).union(rhs_spec).difference(out_spec)
    )

    # reorder lhs and rhs following the order in all indices
    if lhs_spec:
        lhs = ab.transpose(lhs, np.argsort([
            indices_all.index(i) for i in lhs_spec
        ]))
    if rhs_spec:
        rhs = ab.transpose(rhs, np.argsort([
            indices_all.index(i) for i in rhs_spec
        ]))

    # Determine expansion positions to align the contraction, Kronecker,
    # elementwise, and outer product indices
    p_out, p_lhs, p_rhs = 0, 0, 0
    newaxis, colon = None, slice(None)
    index_lhs = []
    index_rhs = []
    ax_contraction = []
    target_shape = []

    for c in indices_all:
        if c not in out_spec:  # contracting index
            ax_contraction.append(p_out)
            index_lhs.append(colon)
            index_rhs.append(colon)
            p_lhs += 1
            p_rhs += 1
            p_out += 1
        else:  # non-contracting index
            if c in kron_spec:
                index_lhs += [colon, newaxis]
                index_rhs += [newaxis, colon]
                target_shape.append(lhs.shape[p_lhs] * rhs.shape[p_rhs])
                p_lhs += 1
                p_rhs += 1
                p_out += 2
            else:
                if c in lhs_spec and c in rhs_spec:
                    target_shape.append(
                        *ab.broadcast_shapes(
                            lhs.shape[p_lhs], rhs.shape[p_rhs]
                        )
                    )
                    index_lhs.append(colon)
                    index_rhs.append(colon)
                    p_lhs += 1
                    p_rhs += 1
                    p_out += 1
                elif c in lhs_spec:
                    index_lhs.append(colon)
                    index_rhs.append(newaxis)
                    target_shape.append(lhs.shape[p_lhs])
                    p_lhs += 1
                    p_out += 1
                elif c in rhs_spec:
                    index_lhs.append(newaxis)
                    index_rhs.append(colon)
                    target_shape.append(rhs.shape[p_rhs])
                    p_rhs += 1
                    p_out += 1

    # compute the contraction in alphabetical order
    op_redu = getattr(ab, reduction)
    op_pair = getattr(ab, pairwise)

    def op_reduce_if(tensor, ax_contraction):
        if ax_contraction:
            return op_redu(tensor, tuple(ax_contraction))
        else:
            return tensor

    return ab.reshape(
        op_reduce_if(
            op_pair(lhs[tuple(index_lhs)], rhs[tuple(index_rhs)]),
            ax_contraction
        ),
        tuple(target_shape),
    )
