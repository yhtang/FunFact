#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._tsrex import TsrEx, EinopEx, _BaseEx


def minplus(lhs: TsrEx, rhs: TsrEx):
    return EinopEx(P.ein(
        _BaseEx(lhs).root, _BaseEx(rhs).root, 6, 'min', 'add', None
    ))


def logsumexp(lhs: TsrEx, rhs: TsrEx):
    return EinopEx(P.ein(
        _BaseEx(lhs).root, _BaseEx(rhs).root, 6, 'log_sum_exp', 'add', None
    ))


def viterbi(lhs: TsrEx, rhs: TsrEx):
    return EinopEx(P.ein(
        _BaseEx(lhs).root, _BaseEx(rhs).root, 6, 'max', 'log_add_exp', None
    ))
