#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab


def _einop(lhs, rhs, spec, output_shape):
    '''Einstein operation between two nd arrays.'''

    if spec.tran_lhs:
        lhs = ab.transpose(lhs, spec.tran_lhs)
    if spec.tran_rhs:
        rhs = ab.transpose(rhs, spec.tran_rhs)

    elementwise = getattr(ab, spec.op_elementwise)(
        lhs[spec.index_lhs], rhs[spec.index_rhs]
    )

    if spec.ax_contraction:
        contracted = getattr(ab, spec.op_reduce)(
            elementwise, spec.ax_contraction
        )
    else:
        contracted = elementwise

    return ab.reshape(contracted, output_shape)
