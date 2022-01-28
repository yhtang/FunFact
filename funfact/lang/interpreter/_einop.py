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

    # determine unsqueeze axes
    # contraction axis, Kronecker indices and remaining indices
    ax_expand_lhs = []
    ax_expand_rhs = []
    ax_contraction = []
    target_shape = []

    p_out, p_lhs, p_rhs = 0, 0, 0
    indices_rem = []

    for c in indices_all:
        if c not in out_spec:  # contracting index
            ax_contraction.append(p_out)
            p_lhs += 1
            p_rhs += 1
            p_out += 1
        else:
            # non-contracting index
            indices_rem.append(c)
            if c in kron_spec:
                ax_expand_lhs.append(p_out)
                ax_expand_rhs.append(p_out + 1)
                target_shape.append(lhs.shape[p_lhs] * rhs.shape[p_rhs])
                p_lhs += 1
                p_rhs += 1
                p_out += 2
            else:
                if c in lhs_spec and c in rhs_spec:
                    target_shape.append(
                        *ab.broadcast_shapes(
                            lhs.shape[p_lhs],
                            rhs.shape[p_rhs]
                        )
                    )
                    p_lhs += 1
                    p_rhs += 1
                    p_out += 1
                elif c in lhs_spec:
                    ax_expand_rhs.append(p_out)
                    target_shape.append(lhs.shape[p_lhs])
                    p_lhs += 1
                    p_out += 1
                elif c in rhs_spec:
                    ax_expand_lhs.append(p_out)
                    target_shape.append(rhs.shape[p_rhs])
                    p_rhs += 1
                    p_out += 1

    # reorder contraction according to out_spec
    final_perm = [indices_rem.index(p_out) for p_out in out_spec]

    print('ax_contraction', ax_contraction)
    print('ax_expand_lhs', ax_expand_lhs)
    print('ax_expand_rhs', ax_expand_rhs)
    print('target_shape', target_shape)
    print('final_perm', final_perm)

    # compute the contraction in alphabetical order
    op_redu = getattr(ab, reduction)
    op_pair = getattr(ab, pairwise)

    def op_reduce_if(tensor, ax_contraction):
        if ax_contraction:
            return op_redu(tensor, tuple(ax_contraction))
        else:
            return tensor

    print(lhs.shape)
    print(ab.expand_dims(lhs, ax_expand_lhs).shape)
    print(ab.expand_dims(rhs, ax_expand_rhs).shape)

    return ab.transpose(
        ab.reshape(
            op_reduce_if(
                op_pair(
                    ab.expand_dims(lhs, ax_expand_lhs),
                    ab.expand_dims(rhs, ax_expand_rhs),
                ),
                ax_contraction
            ),
            tuple(target_shape),
            order='F'
        ),
        final_perm
    )
