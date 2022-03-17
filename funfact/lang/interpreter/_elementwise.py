#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter._evaluation import Evaluator
from ._einop import _einop


def _sliced_size(sli, sz):
    if isinstance(sli, slice):
        return len(range(*sli.indices(sz)))
    elif isinstance(sli, tuple):
        return len(sli)
    else:
        raise RuntimeError(f'Invalid slice {sli}.')


class ElementwiseEvaluator(Evaluator):
    def parametrized_tensor(self, decl, data, slices, **kwargs):
        try:
            return decl.generator(data, slices)
        except TypeError:
            return decl.generator(data)[slices]

    def tensor(self, decl, data, slices, **kwargs):
        return data[slices]

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, einspec,
            shape, slices, **kwargs):
        return _einop(lhs, rhs, einspec, [
            _sliced_size(slice, size) for slice, size in zip(slices, shape)
        ])
